from __future__ import annotations

import operator
import inspect
from contextlib import contextmanager
import jax
import jax.numpy as jnp
import jax.nn as jnn
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type

if TYPE_CHECKING:
    from ._nn_stubs import (
        AxisNamespaceStubs,
        NNBundleStubs,
        NNTensorStubs,
        NNTargetedBundleStubs,
        NNTargetedTensorStubs,
    )
else:
    class AxisNamespaceStubs:
        pass


    class NNTensorStubs:
        pass


    class NNTargetedTensorStubs:
        pass


    class NNBundleStubs:
        pass


    class NNTargetedBundleStubs:
        pass


class CompilerState:
    def __init__(self):
        self.is_initializing = False
        self.params = {}
        # Kept alongside raw arrays so allocation sites can describe physical
        # placement without contaminating JAX values with Python metadata.
        self.param_layouts = {}
        self.param_counter = 0
        self.active_frames = {}
        self.func_calls = {}
        self.tied_scope_override = None
        # Exported/model application is read-only with respect to parameters.
        # This remains separate from ``is_initializing`` so exploratory eager
        # tensor code retains its existing allocation behavior.
        self.strict_params = False
        # ``jax.checkpoint`` introduces implementation frames while tracing a
        # rematerialized function.  Keep its lexical parameter scope explicitly
        # rather than deriving it from those unstable frames.
        self.remat_scope_override = None

    def reset_pass_state(self):
        """Clears the deterministic trackers at the start of a JAX trace."""
        self.param_counter = 0
        self.active_frames.clear()
        self.func_calls.clear()
        self.remat_scope_override = None

    @contextmanager
    def remat_scope(self, scope: str):
        """Make a checkpointed function's parameter scope trace-stable."""
        previous = self.remat_scope_override
        self.remat_scope_override = scope
        try:
            yield
        finally:
            self.remat_scope_override = previous

    def get_scoped_name(self, explicit_name: Optional[str] = None, fallback_prefix: str = "param") -> str:
        """
        Resolves scoping logic:
        1. If tie='@...' -> Global scope.
        2. If tie='...' -> Local execution scope, force this name.
        3. If name='...' -> Semantic label in execution scope, auto-append counter.
        4. No arguments -> Default fallback, auto-append counter.
        """
        if explicit_name is not None and explicit_name.startswith('@'):
            return explicit_name[1:]

        # Tied scan/repeat scopes must remain the outermost ownership boundary.
        # Otherwise a remat scope is explicit and independent of JAX's tracing
        # frames.  Normal eager code keeps the lightweight frame-based name.
        if self.tied_scope_override:
            scope_id = self.tied_scope_override
        elif self.remat_scope_override:
            scope_id = self.remat_scope_override
        else:
            scope_func = "global"
            import sys
            frame = sys._getframe(1)
            try:
                while frame:
                    filepath = frame.f_code.co_filename.replace("\\", "/")
                    if "axiom/" not in filepath and "jax/" not in filepath and "optax/" not in filepath:
                        scope_func = frame.f_code.co_name
                        break
                    frame = frame.f_back
                scope_id = scope_func
            finally:
                # Frame references can retain JAX tracers, so release them at
                # the end of every naming lookup.
                del frame

        if explicit_name:
            return f"{scope_id}/{explicit_name}"
        p_name = f"{scope_id}/{fallback_prefix}_{self.param_counter}"
        self.param_counter += 1
        return p_name

compiler_state = CompilerState()

class Tie:
    def __init__(self, name: str):
        self.name = name


class Axis:
    """Represents a logical dimension in Axiom."""

    def __init__(self, name: str, size: Optional[int] = None, placement=None, *, replicated: bool = False):
        self.name = name
        self.size = size
        if placement is not None and not getattr(placement, "_axiom_mesh_axis", False):
            raise TypeError("Axis placement must be a mesh token such as mesh.tp.")
        if placement is not None and replicated:
            raise ValueError("An axis cannot be both mesh-sharded and explicitly replicated.")
        self.placement = placement
        self.replicated = replicated

    @property
    def layout_explicit(self) -> bool:
        """Whether this axis explicitly requests a layout rather than inference."""
        return self.placement is not None or self.replicated

    def __call__(self, size: Any) -> 'Axis':
        return Axis(self.name, int(size), self.placement, replicated=self.replicated)

    def __getitem__(self, placement) -> 'Axis':
        """Return an axis template explicitly placed on one mesh dimension."""
        if placement is None:
            return Axis(self.name, self.size, replicated=True)
        if not getattr(placement, "_axiom_mesh_axis", False):
            raise TypeError("Axis placement uses mesh.tp or None: ax.d[mesh.tp] or ax.d[None].")
        if self.placement is not None and self.placement != placement:
            raise ValueError(
                f"Axis '{self.name}' is already placed on {self.placement}; it cannot also be placed on {placement}."
            )
        return Axis(self.name, self.size, placement)

    @property
    def mesh_dim(self) -> Optional[str]:
        """Compatibility inspection only; use ``axis.placement`` for new code."""
        return getattr(self.placement, "name", None)

    def shard(self, mesh_dim: str) -> 'Axis':
        raise RuntimeError(
            "Axis.shard(...) has been removed. Create a mesh and write ax.d[mesh.tp](size) instead."
        )

    def __floordiv__(self, other: int) -> int:
        if self.size is None:
            raise ValueError(f"Cannot divide template axis '{self.name}' because it has no size.")
        return self.size // other

    def __mul__(self, other: int) -> int:
        if self.size is None:
            raise ValueError(f"Cannot multiply template axis '{self.name}' because it has no size.")
        return self.size * other

    def __add__(self, other: int) -> int:
        if self.size is None:
            raise ValueError(f"Cannot add to template axis '{self.name}' because it has no size.")
        return self.size + other

    def __repr__(self):
        placement = f"[{self.placement}]" if self.placement is not None else ("[None]" if self.replicated else "")
        return f"Axis({self.name}{placement}={self.size})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Axis) and self.name == other.name


def _resized_axis(axis: Axis, size: Optional[int], *, name: Optional[str] = None, placement=None) -> Axis:
    """Copy an axis while preserving its layout unless an explicit layout is supplied."""
    if placement is None:
        placement = axis.placement
    return Axis(axis.name if name is None else name, size, placement,
                replicated=axis.replicated if placement is None else False)


def _merged_axis(left: Axis, right: Axis) -> Axis:
    """Merge two same-named logical axes and make placement conflicts explicit."""
    if left.name != right.name:
        raise ValueError("Only axes with the same logical name can be aligned.")
    if left.size is not None and right.size is not None and left.size != right.size:
        raise ValueError(f"Axis '{left.name}' has incompatible sizes {left.size} and {right.size}.")
    if left.placement is not None and right.placement is not None and left.placement != right.placement:
        raise ValueError(
            f"Axis '{left.name}' is placed on both {left.placement} and {right.placement}; "
            "make the communication/resharding boundary explicit."
        )
    placed = left.placement or right.placement
    if placed is not None and (left.replicated or right.replicated):
        raise ValueError(
            f"Axis '{left.name}' is explicitly replicated on one operand and sharded on another; "
            "make the resharding boundary explicit."
        )
    return Axis(left.name, left.size if left.size is not None else right.size, placed,
                replicated=left.replicated or right.replicated)


def _decode_parameter_layouts(metadata, mesh):
    """Restore checkpoint layout descriptors against the receiving mesh.

    Checkpoints store mesh *names*, not device identities, which lets a model
    be restored onto an equivalent fresh mesh (including a CPU test mesh).
    """
    from .layout import ParameterLayout

    def decode_axes(entries):
        result = []
        for entry in entries:
            placement_name = entry.get("placement")
            if placement_name is not None and mesh is None:
                raise ValueError(
                    "Checkpoint contains placed parameters; load into a model constructed with the intended ax.mesh(...)."
                )
            placement = getattr(mesh, placement_name) if placement_name is not None else None
            result.append(Axis(entry["name"], entry.get("size"), placement,
                               replicated=entry.get("replicated", False)))
        return tuple(result)

    return {
        name: ParameterLayout(
            axes=decode_axes(entry["axes"]),
            kind=entry.get("kind", "tensor"),
            input_axes=decode_axes(entry.get("input_axes", ())),
            output_axes=decode_axes(entry.get("output_axes", ())),
        )
        for name, entry in metadata.get("parameters", {}).items()
    }


class _AxisNamespace(AxisNamespaceStubs):
    def __call__(self, name: str, size: Optional[int] = None) -> Axis:
        return Axis(name, size)

    def model(self, fn, params=None, *, mesh=None) -> 'AxiomModel':
        from .compiler import AxiomModel
        return AxiomModel(fn, params, mesh=mesh)

    def mesh(self, *, devices=None, **axis_sizes):
        """Create an explicit device mesh used by axis placement annotations."""
        from .layout import AxiomMesh
        return AxiomMesh(devices=devices, **axis_sizes)

    @property
    def jit(self):
        from .compiler import jit
        return jit

    @property
    def value_and_grad(self):
        from .compiler import value_and_grad
        return value_and_grad

    @property
    def grad(self):
        from .compiler import grad
        return grad

    @property
    def apply_updates(self):
        from .compiler import apply_updates
        return apply_updates

    @property
    def to_jax(self):
        from .compiler import to_jax
        return to_jax

    def stack(self, items: list[Any], new_axis: 'Axis') -> Any:
        if not items:
            raise ValueError("Cannot stack an empty list.")

        # Recursive Bundle Stacking
        if hasattr(items[0], 'tensors'):  # Duck-typing for Bundle
            num_inner = len(items[0].tensors)
            stacked_tensors = []
            for i in range(num_inner):
                # Extract the i-th tensor across all bundles and stack them
                inner_list = [bundle.tensors[i] for bundle in items]
                stacked_tensors.append(self.stack(inner_list, new_axis))
            return Bundle(*stacked_tensors)

        base_top = items[0].topology
        for t in items:
            if tuple(axis.name for axis in t.topology) != tuple(axis.name for axis in base_top):
                raise ValueError(f"Topology mismatch in stack: expected {[a.name for a in base_top]}, got {[a.name for a in t.topology]}.")
            base_top = tuple(_merged_axis(left, right) for left, right in zip(base_top, t.topology))

        import jax.numpy as jnp
        raw_arrays = [t.unwrap() for t in items]
        stacked_raw = jnp.stack(raw_arrays, axis=0)
        return wrap(stacked_raw, new_axis, *base_top)

    def save(self, target: Any, path: str):
        """Save parameters plus portable Axiom layout metadata when available."""
        import json
        import numpy as np

        # Extract params whether it's an AxiomModel or a raw dict
        params = target.params if hasattr(target, 'params') else target

        if not isinstance(params, dict) or not params:
            raise ValueError("Target has no parameters to save.")

        numpy_params = {k: np.array(v) for k, v in params.items()}
        if hasattr(target, "param_layouts"):
            from .layout import AxiomLayout
            if getattr(target, "mesh", None) is not None:
                metadata = target.layout.metadata()
            else:
                metadata = {
                    "mesh_axes": {},
                    "parameters": {name: layout.metadata() for name, layout in target.param_layouts.items()},
                }
            numpy_params["__axiom_layout__"] = np.asarray(json.dumps(metadata))

        if not path.endswith('.npz'):
            path += '.npz'

        np.savez_compressed(path, **numpy_params)
        print(f"Saved {len(params)} parameters to {path}")

    def load(self, path: str, *, target: Any = None):
        """
        Loads parameters from disk.
        If 'target' is an AxiomModel, it injects the weights directly.
        If 'target' is None, it returns the raw JAX parameter dictionary.
        """
        import json
        import numpy as np
        import jax.numpy as jnp

        if not path.endswith('.npz'):
            path += '.npz'

        loaded = np.load(path)
        metadata = None
        if "__axiom_layout__" in loaded.files:
            metadata = json.loads(str(loaded["__axiom_layout__"].item()))
        params_dict = {k: jnp.array(loaded[k]) for k in loaded.files if k != "__axiom_layout__"}

        # Paradigm 1: AxiomModel Injection
        if target is not None:
            if hasattr(target, 'params'):
                target.params = params_dict
                if not getattr(target, "param_layouts", None) and metadata is not None:
                    target.param_layouts = _decode_parameter_layouts(metadata, getattr(target, "mesh", None))
                if getattr(target, "mesh", None) is not None:
                    target.params = target.layout.place_params(target.params)
                target.is_initialized = True
                print(f"Loaded {len(target.params)} parameters into model from {path}")
                return
            else:
                raise TypeError("Target must be an AxiomModel or None.")

        # Paradigm 2: Pure JAX Dictionary Return
        print(f"Loaded {len(params_dict)} parameters from {path}")
        return params_dict

    def trace(self, *topology: 'Axis', dtype=None, max_int=32000, init_fn=None):
        """
        Eagerly executes a block with synthesized data to trigger IDE breakpoints.

        Usage:
            # Default normal distribution
            @ax.trace(ax.b(4), ax.s(1024), ax.d(128))

            # Custom deterministic override
            @ax.trace(ax.b(1), ax.s(32), init_fn=init.ones)
        """
        import jax.numpy as jnp
        from . import init

        dtype = dtype or jnp.float32

        def decorator(func):
            print(f"\n🔬 Axiom Trace: Triggering breakpoints for '{func.__name__}'...")

            # 1. Synthesize data: Custom vs. Heuristic
            if init_fn is not None:
                # Use the user's explicit initialization
                dummy_input = init_fn(*topology).astype(dtype)
            else:
                # Fallback to smart heuristics
                if jnp.issubdtype(jnp.dtype(dtype), jnp.integer):
                    dummy_input = (init.uniform(*topology) * max_int).astype(dtype)
                else:
                    dummy_input = init.normal(*topology).astype(dtype)

            # 2. Execute eagerly through the model wrapper
            from .compiler import AxiomModel
            model = AxiomModel(func)

            # 3. Hit the breakpoint natively!
            out = model(dummy_input)

            topo_str = ", ".join([f"{a.name}[{a.size}]" for a in out.topology])
            print(f"✅ Trace complete. Output: Tensor[{topo_str}]")

            return func

        return decorator

    @property
    def remat(self):
        """Gradient checkpointing with deterministic parameter ownership.

        A checkpointed function is retraced by JAX during its backward pass.
        Its parameters therefore use an explicit lexical scope while the body
        runs, rather than inferring one from JAX's transient Python frames.
        This keeps names, sharding metadata, tied parameters, and checkpoints
        identical in eager, ``jit``, and reverse-mode execution.
        """
        import jax
        from functools import wraps

        def wrapper(func):
            # ``__qualname__`` distinguishes two independently defined local
            # ``block`` functions while remaining stable when a model factory
            # is reconstructed for checkpoint restore.
            scope = func.__qualname__

            # A pure inner function that explicitly accepts the parameter dictionary
            def pure_func(params_dict, *args, **kwargs):
                # 1. Swap the global state to the explicitly tracked JAX parameters
                prev_params = getattr(compiler_state, 'params', {})
                compiler_state.params = params_dict

                try:
                    with compiler_state.remat_scope(scope):
                        res = func(*args, **kwargs)
                finally:
                    # 2. Safely restore
                    compiler_state.params = prev_params

                return res

            @wraps(func)
            def inner(*args, **kwargs):
                with compiler_state.remat_scope(scope):
                    if compiler_state.is_initializing:
                        # Ghost Pass: allocate eagerly, with the exact same
                        # scope that the later checkpoint trace will use.
                        return func(*args, **kwargs)
                    # Explicitly pass the global params through the checkpoint
                    # boundary so JAX can rematerialize without residual leaks.
                    return jax.checkpoint(pure_func)(compiler_state.params, *args, **kwargs)

            return inner

        return wrapper

    @property
    def grid(self):
        """Current Pallas program coordinates, indexed by named Axis."""
        from .kernel import grid
        return grid

    @property
    def tile(self):
        """Current Pallas tile-local register indices, indexed by named Axis."""
        from .kernel import tile
        return tile

    def __getattr__(self, name: str) -> Axis:
        return Axis(name)


ax = _AxisNamespace()


class SlicedMonad:
    """Lazy slice / optional patch transaction."""

    def __init__(self, original_tensor, target_ax, slice_obj, chunk_tensor, expected_topology=None,
                 patch_safe: bool = True, unsafe_reason: Optional[str] = None):
        self.original_tensor = original_tensor
        self.target_ax = target_ax
        self.slice_obj = slice_obj
        self.chunk_tensor = chunk_tensor
        self.expected_topology = tuple(expected_topology or chunk_tensor.topology)
        self.patch_safe = patch_safe
        self.unsafe_reason = unsafe_reason

    def _as_tensor(self):
        return self.chunk_tensor

    def unwrap(self):
        return self.chunk_tensor.unwrap()

    @property
    def topology(self):
        return self.chunk_tensor.topology

    def _topology_signature(self, topology):
        return tuple((a.name, a.size) for a in topology)

    def _current_target_ax(self):
        for a in self.chunk_tensor.topology:
            if a.name == self.target_ax.name:
                return a
        raise ValueError(f"Original patch axis '{self.target_ax.name}' is no longer in topology.")

    def _wrap(self, result, *, patch_safe: Optional[bool] = None, unsafe_reason: Optional[str] = None):
        if hasattr(result, "chunk_tensor"): result = result.chunk_tensor
        if not isinstance(result, Tensor): return result

        if patch_safe is None: patch_safe = self.patch_safe
        if unsafe_reason is None: unsafe_reason = self.unsafe_reason

        return SlicedMonad(self.original_tensor, self.target_ax, self.slice_obj, result,
                           expected_topology=self.expected_topology, patch_safe=patch_safe, unsafe_reason=unsafe_reason)

    def pw(self, func, **kwargs) -> 'SlicedMonad':
        return self._wrap(TargetedTensor(self.chunk_tensor, (self._current_target_ax(),)).pw(func, **kwargs))

    def proj(self, *target_axes, bias: bool = True, tie: Optional[str] = None, init=None) -> 'SlicedMonad':
        new_chunk = TargetedTensor(self.chunk_tensor, (self._current_target_ax(),)).proj(*target_axes, bias=bias,
                                                                                         tie=tie, init=init)
        if len(target_axes) > 0:
            return self._wrap(new_chunk, patch_safe=False, unsafe_reason="Explicit proj(...) is not commit-safe.")
        return self._wrap(new_chunk)

    def bias(self, init=None, tie: Optional[str] = None) -> 'SlicedMonad':
        return self._wrap(TargetedTensor(self.chunk_tensor, (self._current_target_ax(),)).bias(init=init, tie=tie))

    def gate(self, init=None, tie: Optional[str] = None) -> 'SlicedMonad':
        return self._wrap(TargetedTensor(self.chunk_tensor, (self._current_target_ax(),)).gate(init=init, tie=tie))

    def __getattr__(self, name):
        for axis in self.chunk_tensor.topology:
            if axis.name == name:
                return TargetedSlicedMonad(self, (axis,))

        attr = getattr(self.chunk_tensor, name)
        if callable(attr):
            def wrapped(*args, **kwargs): return self._wrap(attr(*args, **kwargs))

            return wrapped
        return attr

    def _unwrap_other(self, other):
        if hasattr(other, "chunk_tensor"): return other.chunk_tensor
        if hasattr(other, "tensor") and hasattr(other, "target_axes"): return other.tensor
        return other

    def _binary_value_or_patch(self, other, op):
        other_is_monad = hasattr(other, "chunk_tensor")
        result = op(self.chunk_tensor, self._unwrap_other(other))
        return result if other_is_monad else self._wrap(result)

    def _rbinary_value_or_patch(self, other, op):
        other_is_monad = hasattr(other, "chunk_tensor")
        result = op(self._unwrap_other(other), self.chunk_tensor)
        return result if other_is_monad else self._wrap(result)

    def __add__(self, other):
        return self._binary_value_or_patch(other, operator.add)

    def __radd__(self, other):
        return self._rbinary_value_or_patch(other, operator.add)

    def __sub__(self, other):
        return self._binary_value_or_patch(other, operator.sub)

    def __rsub__(self, other):
        return self._rbinary_value_or_patch(other, operator.sub)

    def __mul__(self, other):
        return self._binary_value_or_patch(other, operator.mul)

    def __rmul__(self, other):
        return self._rbinary_value_or_patch(other, operator.mul)

    def __truediv__(self, other):
        return self._binary_value_or_patch(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._rbinary_value_or_patch(other, operator.truediv)

    def __neg__(self):
        return self._wrap(-self.chunk_tensor)

    def __pow__(self, other):
        return self._binary_value_or_patch(other, operator.pow)

    def __rpow__(self, other):
        return self._rbinary_value_or_patch(other, operator.pow)

    def __matmul__(self, other):
        return self.chunk_tensor @ self._unwrap_other(other)

    def __and__(self, other):
        return self.chunk_tensor & self._unwrap_other(other)

    def __getitem__(self, key) -> 'Tensor':
        if key != slice(None):
            raise ValueError("Use [:] to stitch the sliced monad back into the parent tensor.")
        if not self.patch_safe:
            raise ValueError(f"Cannot commit unsafe patch. {self.unsafe_reason or ''}")

        expected_sig = self._topology_signature(self.expected_topology)
        actual_sig = self._topology_signature(self.chunk_tensor.topology)
        if actual_sig != expected_sig:
            # Reverted to match your lowercase 'topology' regex check
            raise ValueError(f"Cannot commit sliced patch because the chunk topology changed from {expected_sig} to {actual_sig}. Stitch manually.")

        # Restored the strict contiguous check
        if not isinstance(self.slice_obj, slice):
            raise ValueError("Sliced patch commit currently only supports Python slice objects.")
        if self.slice_obj.step not in (None, 1):
            raise ValueError("Sliced patch commit currently only supports contiguous slices with step None or 1.")

        orig_raw = self.original_tensor.unwrap()
        chunk_raw = self.chunk_tensor.unwrap()
        ax_idx = self.original_tensor.topology.index(self.target_ax)
        start, stop, step = self.slice_obj.indices(orig_raw.shape[ax_idx])

        left_slice = [slice(None)] * orig_raw.ndim
        right_slice = [slice(None)] * orig_raw.ndim
        left_slice[ax_idx] = slice(None, start)
        right_slice[ax_idx] = slice(stop, None)

        stitched_raw = jnp.concatenate([orig_raw[tuple(left_slice)], chunk_raw, orig_raw[tuple(right_slice)]], axis=ax_idx)
        return Tensor(stitched_raw, *self.original_tensor.topology)


class TargetedSlicedMonad:
    """Targeted view into a SlicedMonad's chunk, preserving patch context."""

    def __init__(self, monad: SlicedMonad, target_axes: Tuple[Axis, ...]):
        self.monad = monad
        self.target_axes = target_axes

    @property
    def tensor(self):
        return self.monad.chunk_tensor

    def _targeted(self):
        return TargetedTensor(self.monad.chunk_tensor, self.target_axes)

    def _wrap(self, result, *, patch_safe: Optional[bool] = None, unsafe_reason: Optional[str] = None):
        return self.monad._wrap(result, patch_safe=patch_safe, unsafe_reason=unsafe_reason)

    def __getattr__(self, name: str):
        for axis in self.monad.chunk_tensor.topology:
            if axis.name == name:
                if axis not in self.target_axes:
                    return TargetedSlicedMonad(self.monad, self.target_axes + (axis,))
                return self

        attr = getattr(self._targeted(), name)
        if callable(attr):
            def wrapped(*args, **kwargs): return self._wrap(attr(*args, **kwargs))

            return wrapped
        return attr

    def pw(self, func, tie: Optional[str] = None, **kwargs):
        return self._wrap(self._targeted().pw(func, tie=tie, **kwargs))

    def bias(self, init=None, tie: Optional[str] = None):
        return self._wrap(self._targeted().bias(init=init, tie=tie))

    def gate(self, init=None, tie: Optional[str] = None):
        return self._wrap(self._targeted().gate(init=init, tie=tie))

    def sum(self):
        return self._targeted().sum()

    def mean(self):
        return self._targeted().mean()

    def max(self):
        return self._targeted().max()

    def rename(self, new_axis: Axis):
        return self._wrap(self._targeted().rename(new_axis), patch_safe=False,
                          unsafe_reason="rename(...) inside a sliced patch is not commit-safe.")

    def pad(self, *pad_widths, fill: float = 0.0):
        return self._wrap(self._targeted().pad(*pad_widths, fill=fill), patch_safe=False,
                          unsafe_reason="pad(...) inside a sliced patch is not commit-safe.")

    def unfold(self, window_axis: Axis, step: int = 1):
        return self._wrap(self._targeted().unfold(window_axis, step=step), patch_safe=False,
                          unsafe_reason="unfold(...) inside a sliced patch is not commit-safe.")

    @property
    def size(self) -> int:
        return self._targeted().size

    def __int__(self) -> int:
        return int(self._targeted())

    def __index__(self) -> int:
        return self._targeted().__index__()

    def __getitem__(self, key):
        return self._wrap(self._targeted().__getitem__(key))

    def __matmul__(self, other):
        return self._targeted().__matmul__(other.chunk_tensor if hasattr(other, "chunk_tensor") else other)


class TargetedTensor(NNTargetedTensorStubs):
    """The transient object returned when targeting axes (e.g., x.d)"""

    def __init__(self, tensor: 'Tensor', target_axes: Tuple[Axis, ...]):
        self.tensor = tensor
        self.target_axes = target_axes

    @property
    def topology(self):
        """Expose the annotated tensor topology for inspection/export helpers."""
        return self.tensor.topology

    def unwrap(self):
        return self.tensor.unwrap()

    def __call__(self, tile_size: int):
        """Turn a single named target into a tile-aware Pallas axis reference."""
        if len(self.target_axes) != 1:
            raise ValueError("Tile one axis at a time, e.g. tensor.b(1).s(128).map(fn).")
        from .kernel import tile_axis
        return tile_axis(self.tensor, self.target_axes[0].name, tile_size)

    def __getattr__(self, name: str) -> Any:
        """The Universal Dispatcher: Chains axes or dynamically invokes pure JAX/NN primitives."""
        # 1. Axis Targeting
        for axis in self.tensor.topology:
            if axis.name == name:
                if axis not in self.target_axes:
                    return TargetedTensor(self.tensor, self.target_axes + (axis,))
                return self

        # 2. Dynamic Universal Dispatcher
        from . import nn

        # Is it an Axiom NN module?
        if hasattr(nn, name) and callable(getattr(nn, name)) and not name.endswith('_loss'):
            return lambda *args, **kwargs: self.pw(getattr(nn, name), *args, **kwargs)

        # Is it a JAX primitive?
        func = getattr(jnp, name, None) or getattr(jnn, name, None)
        if callable(func):
            return lambda *args, **kwargs: self.pw(func, *args, **kwargs)

        raise AttributeError(f"Targeted axis, NN function, or JAX primitive '{name}' not found.")

    def pw(self, func, *args, tie: Optional[str] = None, **kwargs) -> 'Tensor':
        """Executes a function while preserving topology and intelligently injecting the 'axis' parameter."""
        if getattr(func, '_is_axiom_nn', False):
            if tie is None:
                return func(self, *args, **kwargs)
            return func(self, *args, tie=tie, **kwargs)

        try:
            sig = inspect.signature(func)
            if 'axis' in sig.parameters:
                axis_indices = tuple(self.tensor.topology.index(a) for a in self.target_axes)
                kwargs['axis'] = axis_indices if len(axis_indices) > 1 else axis_indices[0]
        except ValueError:
            pass

        raw_result = func(self.tensor.unwrap(), *args, **kwargs)

        # --- Auto-detect Dimensionality Reduction ---
        if hasattr(raw_result, 'shape') and len(raw_result.shape) < len(self.tensor.topology):
            new_topology = tuple(a for a in self.tensor.topology if a not in self.target_axes)
            return Tensor(raw_result, *new_topology)

        return Tensor(raw_result, *self.tensor.topology)

    # --- TOPOLOGICAL MUTATORS & PARAMETER ALLOCATORS (The Explicit overrides) ---

    def proj(self, *target_axes: 'Axis', bias: bool = True, tie: Optional[str] = None, init=None) -> 'Tensor':
        if not target_axes:
            target_axes = self.target_axes
        else:
            # Auto-infer missing sizes from existing topology
            resolved_axes = []
            for tgt_ax in target_axes:
                if tgt_ax.size is None:
                    # Look for it in the current tensor's topology
                    for current_ax in self.tensor.topology:
                        if current_ax.name == tgt_ax.name:
                            resolved_axes.append(
                                Axis(tgt_ax.name, current_ax.size, tgt_ax.placement,
                                     replicated=tgt_ax.replicated)
                                if tgt_ax.layout_explicit else current_ax
                            )
                            break
                    else:
                        raise ValueError(f"Cannot project into '{tgt_ax.name}' without a size.")
                else:
                    resolved_axes.append(tgt_ax)
            target_axes = tuple(resolved_axes)

        import numpy as np
        from .state import state
        from . import init as ax_init

        in_dim = np.prod([a.size for a in self.target_axes])
        out_dim = np.prod([a.size for a in target_axes])
        initializer = init if init is not None else ax_init.xavier

        from .layout import ParameterLayout
        weight_layout = ParameterLayout(
            axes=(Axis("_in", in_dim), Axis("_out", out_dim)),
            kind="projection",
            input_axes=tuple(self.target_axes),
            output_axes=tuple(target_axes),
        )
        W_raw = state.get_param(
            "proj_w", (in_dim, out_dim), initializer, tie=tie, fan_in=in_dim, fan_out=out_dim,
            layout=weight_layout,
        )
        W_param = Tensor(W_raw, Axis("_in", in_dim), Axis("_out", out_dim))

        kept_axes = tuple(a for a in self.tensor.topology if a not in self.target_axes)
        transpose_order = [self.tensor.topology.index(a) for a in kept_axes + self.target_axes]
        transposed_raw = jnp.transpose(self.tensor.unwrap(), transpose_order)

        kept_shape = tuple(a.size if a.size is not None else -1 for a in kept_axes)
        flattened_raw = jnp.reshape(transposed_raw, kept_shape + (in_dim,))

        result_raw = jnp.dot(flattened_raw, W_param.unwrap())

        new_topology = kept_axes + target_axes
        new_shape = tuple(a.size if a.size is not None else -1 for a in new_topology)
        result_tensor = Tensor(jnp.reshape(result_raw, new_shape), *new_topology)

        if bias:
            return TargetedTensor(result_tensor, target_axes).bias(tie=f"{tie}_bias" if tie else None)

        return result_tensor

    def bias(self, init=None, tie: Optional[str] = None) -> 'Tensor':
        from .state import state
        from . import init as ax_init
        initializer = init if init is not None else ax_init.zeros
        shape = tuple(a.size for a in self.target_axes)
        from .layout import ParameterLayout
        b_raw = state.get_param(
            "bias", shape, initializer, tie=tie,
            layout=ParameterLayout(tuple(self.target_axes)),
        )
        return self.tensor + Tensor(b_raw, *self.target_axes)

    def gate(self, init=None, tie: Optional[str] = None) -> 'Tensor':
        from .state import state
        from . import init as ax_init
        initializer = init if init is not None else ax_init.ones
        shape = tuple(a.size for a in self.target_axes)
        from .layout import ParameterLayout
        g_raw = state.get_param(
            "gate", shape, initializer, tie=tie,
            layout=ParameterLayout(tuple(self.target_axes)),
        )
        return self.tensor * Tensor(g_raw, *self.target_axes)

    def mask(self, func, fill: float) -> 'Tensor':
        grids = [jnp.arange(a.size) for a in self.target_axes]
        bool_mask = func(*jnp.meshgrid(*grids, indexing='ij'))
        aligned_mask = Tensor(bool_mask, *self.target_axes)._align_to(self.tensor.topology)
        return Tensor(jnp.where(aligned_mask, fill, self.tensor.unwrap()), *self.tensor.topology)

    def vmask(self, func, fill: float = 0.0) -> 'Tensor':
        return Tensor(jnp.where(func(self.tensor.unwrap()), fill, self.tensor.unwrap()), *self.tensor.topology)

    def merge(self, new_axis: Axis) -> 'Tensor':
        """
        Pure topological flattening.
        Merges all targeted axes into a single new axis.
        e.g., x.h.w.merge(ax.s)
        """
        import jax.numpy as jnp
        import numpy as np

        sizes = [a.size for a in self.target_axes]
        if None in sizes:
            raise ValueError("Cannot merge axes with undefined sizes.")
        total_size = int(np.prod(sizes))

        if new_axis.size is not None and new_axis.size != total_size:
            raise ValueError(
                f"Topological Violation: Cannot merge axes of sizes {sizes} (total: {total_size}) "
                f"into '{new_axis.name}' of size {new_axis.size}."
            )

        source_placements = {axis.placement for axis in self.target_axes if axis.placement is not None}
        if source_placements and new_axis.placement is None and not new_axis.replicated:
            raise ValueError(
                f"Merging placed axis/axes {[axis.name for axis in self.target_axes]} requires an explicit "
                f"target placement, e.g. ax.{new_axis.name}[mesh.<axis>](...)."
            )
        final_axis = Axis(new_axis.name, total_size, new_axis.placement, replicated=new_axis.replicated)

        kept_axes = tuple(a for a in self.tensor.topology if a not in self.target_axes)
        transpose_order = [self.tensor.topology.index(a) for a in kept_axes + self.target_axes]

        transposed_raw = jnp.transpose(self.tensor.unwrap(), transpose_order)

        kept_shape = tuple(a.size if a.size is not None else -1 for a in kept_axes)
        merged_raw = jnp.reshape(transposed_raw, kept_shape + (total_size,))

        new_topology = kept_axes + (final_axis,)
        return Tensor(merged_raw, *new_topology)

    def split(self, *new_axes: Axis) -> 'Tensor':
        """
        Pure topological splitting.
        Splits a single targeted axis into multiple new axes, supporting size inference.
        e.g., x.d.split(ax.heads(4), ax.head_dim)
        """
        if len(self.target_axes) != 1:
            raise ValueError("split() requires exactly one targeted axis. Merge first if splitting multiple.")

        target_ax = self.target_axes[0]

        import jax.numpy as jnp

        known_size = 1
        unknown_axes = []
        for a in new_axes:
            if a.size is None:
                unknown_axes.append(a)
            else:
                known_size *= a.size

        if len(unknown_axes) > 1:
            raise ValueError("Can only infer the size of one axis during split().")

        final_new_axes = list(new_axes)
        if unknown_axes:
            if target_ax.size is None:
                raise ValueError(
                    f"Cannot infer size for '{unknown_axes[0].name}' because target axis '{target_ax.name}' has no size.")
            if target_ax.size % known_size != 0:
                raise ValueError(f"Cannot cleanly divide {target_ax.size} by {known_size} for axis inference.")

            inferred_size = target_ax.size // known_size

            # Replace the unknown axis with a strictly sized one
            for i, a in enumerate(final_new_axes):
                if a.size is None:
                    final_new_axes[i] = _resized_axis(a, inferred_size)
        else:
            if target_ax.size is not None and known_size != target_ax.size:
                raise ValueError(
                    f"Topological Violation: Cannot split axis of size {target_ax.size} into {known_size}.")

        if target_ax.placement is not None:
            placed_outputs = [axis for axis in final_new_axes if axis.placement == target_ax.placement]
            if len(placed_outputs) != 1:
                raise ValueError(
                    f"Splitting placed axis '{target_ax.name}' requires exactly one output axis placed on "
                    f"{target_ax.placement}, e.g. ax.h[{target_ax.placement}](...)."
                )

        ax_idx = self.tensor.topology.index(target_ax)
        raw_shape = self.tensor.unwrap().shape

        new_shape = raw_shape[:ax_idx] + tuple(a.size for a in final_new_axes) + raw_shape[ax_idx + 1:]
        split_raw = jnp.reshape(self.tensor.unwrap(), new_shape)

        new_topology = self.tensor.topology[:ax_idx] + tuple(final_new_axes) + self.tensor.topology[ax_idx + 1:]

        return Tensor(split_raw, *new_topology)

    def unfold(self, window_axis: Axis, step: int = 1) -> 'Tensor':
        spatial_ax = self.target_axes[0]
        out_size = (spatial_ax.size - window_axis.size) // step + 1

        starts = jnp.arange(0, out_size * step, step)[:, None]
        offsets = jnp.arange(window_axis.size)[None, :]

        ax_idx = self.tensor.topology.index(spatial_ax)
        unfolded_raw = jnp.take(self.tensor.unwrap(), starts + offsets, axis=ax_idx)

        new_topology = list(self.tensor.topology)
        new_topology.pop(ax_idx)
        new_topology.insert(ax_idx, window_axis)
        new_topology.insert(ax_idx, _resized_axis(spatial_ax, out_size))

        return Tensor(unfolded_raw, *new_topology)

    def scan(self, func: callable, init: 'Tensor'):
        import jax.numpy as jnp
        import jax.lax as lax
        from .compiler import compiler_state

        target_ax = self.target_axes[0]
        base_tensor = self.tensor  # <-- THE FIX: Extract the underlying Tensor!
        seq_idx = base_tensor.topology.index(target_ax)

        # --- 1. Weight Tying Scope Setup ---
        scan_scope = f"scan_block_{compiler_state.param_counter}"
        prev_override = compiler_state.tied_scope_override
        start_counter = compiler_state.param_counter

        # --- 2. The Ghost Pass Bypass ---
        if compiler_state.is_initializing:
            compiler_state.tied_scope_override = scan_scope

            # Use base_tensor instead of self!
            raw_slice = jnp.take(base_tensor.unwrap(), 0, axis=seq_idx)
            slice_top = tuple(a for a in base_tensor.topology if a != target_ax)
            mock_x = Tensor(raw_slice, *slice_top)

            _ = func(init, mock_x)

        # --- 3. The Trace Pass ---
        compiler_state.tied_scope_override = scan_scope

        # Route all permutations through base_tensor
        perm = [seq_idx] + [i for i in range(len(base_tensor.topology)) if i != seq_idx]
        xs_transposed = jnp.transpose(base_tensor.unwrap(), axes=perm)
        xt_topology = tuple(base_tensor.topology[i] for i in perm[1:])

        def wrapped_func(raw_carry, raw_xt):
            compiler_state.param_counter = start_counter
            new_carry, out_y = func(Tensor(raw_carry, *init.topology), Tensor(raw_xt, *xt_topology))
            return new_carry.unwrap(), out_y.unwrap()

        final_carry_raw, y_seq_raw = lax.scan(wrapped_func, init.unwrap(), xs_transposed)

        # --- 4. Cleanup & State Restoration ---
        compiler_state.tied_scope_override = prev_override
        allocated = len([k for k in compiler_state.params if k.startswith(scan_scope)])
        compiler_state.param_counter = start_counter + allocated

        inv_perm = [0] * len(base_tensor.topology)
        for i, p in enumerate(perm):
            inv_perm[p] = i

        return Tensor(final_carry_raw, *init.topology), Tensor(jnp.transpose(y_seq_raw, inv_perm),
                                                               *base_tensor.topology)

    def sample(self, temp: float = 1.0) -> 'Tensor':
        from .state import state
        class_idx = self.tensor.topology.index(self.target_axes[0])
        out_topology = tuple(a for a in self.tensor.topology if a not in self.target_axes)
        logits_raw = self.tensor.unwrap()

        if temp == 0.0:
            sampled_raw = jnp.argmax(logits_raw, axis=class_idx)
        else:
            sampled_raw = jax.random.categorical(state.next_key(), logits_raw / temp, axis=class_idx)

        return Tensor(sampled_raw, *out_topology)

    def pad(self, *pad_widths: Tuple[int, int], fill: float = 0.0) -> 'Tensor':
        if len(pad_widths) == 1: pad_widths = pad_widths * len(self.target_axes)

        pad_width_full = [(0, 0)] * len(self.tensor.topology)
        new_topology = list(self.tensor.topology)

        for target, (before, after) in zip(self.target_axes, pad_widths):
            ax_idx = self.tensor.topology.index(target)
            pad_width_full[ax_idx] = (before, after)
            new_topology[ax_idx] = _resized_axis(target, (target.size + before + after) if target.size else None)

        return Tensor(jnp.pad(self.tensor.unwrap(), pad_width_full, constant_values=fill), *new_topology)

    def rename(self, new_axis: Axis) -> 'Tensor':
        target = self.target_axes[0]
        if new_axis.size is not None and new_axis.size != target.size:
            raise ValueError(f"Topological Violation: .rename() cannot change axis sizes.")

        new_topology = list(self.tensor.topology)
        replacement = new_axis if new_axis.layout_explicit else target
        new_topology[self.tensor.topology.index(target)] = Axis(
            new_axis.name, target.size, replacement.placement, replicated=replacement.replicated
        )
        return Tensor(self.tensor.unwrap(), *new_topology)

    def _reduce(self, jnp_func) -> 'Tensor':
        axis_indices = tuple(self.tensor.topology.index(a) for a in self.target_axes)
        new_topology = tuple(a for a in self.tensor.topology if a not in self.target_axes)
        return Tensor(jnp_func(self.tensor.unwrap(), axis=axis_indices), *new_topology)

    def sum(self) -> 'Tensor':
        return self._reduce(jnp.sum)

    def mean(self) -> 'Tensor':
        return self._reduce(jnp.mean)

    def max(self) -> 'Tensor':
        return self._reduce(jnp.max)

    @property
    def size(self) -> int:
        import numpy as np
        return int(np.prod([a.size for a in self.target_axes]))

    def __int__(self) -> int:
        return self.size

    def __index__(self) -> int:
        return self.size

    def __mul__(self, other):
        return self.size * other if isinstance(other, int) else self.tensor * other

    def __rmul__(self, other):
        return self.tensor * other

    def __floordiv__(self, other):
        return self.size // other if isinstance(other, int) else self.tensor // other

    def __add__(self, other):
        return self.size + other if isinstance(other, int) else self.tensor + other

    def __radd__(self, other):
        return self.tensor + other

    def __sub__(self, other):
        return self.tensor - other

    def __rsub__(self, other):
        return other - self.tensor

    def __truediv__(self, other):
        return self.tensor / other

    def __rtruediv__(self, other):
        return other / self.tensor

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if not hasattr(other, 'topology'):
            raise ValueError("Matrix multiplication requires another Tensor.")

        for target_ax in self.target_axes:
            if target_ax not in other.topology:
                raise ValueError(f"Cannot contract over '{target_ax.name}': axis missing in right tensor.")

        # Native broadcasting multiplication
        result = self.tensor * other

        # Sum over ONLY the targeted axes
        for target_ax in self.target_axes:
            result = getattr(result, target_ax.name).sum()

        return result

    def __iter__(self):
        """Allows Pythonic unpacking of a targeted axis: x1, x2 = x.split_ax"""
        if len(self.target_axes) > 1:
            raise ValueError("Can only unpack a single targeted axis at a time.")

        target_ax = self.target_axes[0]

        if target_ax.size is None:
            raise ValueError(f"Cannot unpack axis '{target_ax.name}' because its size is unknown.")

        for i in range(target_ax.size):
            yield self[i]

    def __getitem__(self, item: Any) -> Any:
        """Allows shape-safe slicing, integer indexing, and routed gathering."""
        import jax.numpy as jnp

        # Placement has priority over numerical indexing so the two compose:
        # ``x.d[mesh.tp][:128]`` is a placed view followed by an ordinary
        # Axiom slice, while bare ``x.d[:128]`` remains unchanged.
        if item is None or getattr(item, "_axiom_mesh_axis", False):
            if len(self.target_axes) != 1:
                raise ValueError("Place one logical axis at a time, e.g. x.d[mesh.tp] or x.d[None].")
            target_ax = self.target_axes[0]
            if item is not None and target_ax.placement is not None and target_ax.placement != item:
                raise ValueError(
                    f"Axis '{target_ax.name}' is already placed on {target_ax.placement}; cannot assert {item}."
                )
            placed_axis = Axis(target_ax.name, target_ax.size, item, replicated=item is None)
            placed_topology = tuple(placed_axis if axis == target_ax else axis for axis in self.tensor.topology)
            placed_tensor = Tensor(self.tensor.unwrap(), *placed_topology)
            return TargetedTensor(placed_tensor, (placed_axis,))

        # 1. Monad Stitching Fallback
        if isinstance(item, slice) and item == slice(None):
            return self.tensor

        # 2. Routed Context (Gathering)
        # If the index is another Tensor, intercept it for .gather()!
        if hasattr(item, 'topology'):
            return RoutedContext(self.tensor, self.target_axes, item)

        # Decay monads if passed as an index
        if hasattr(item, "chunk_tensor"):
            item = item.chunk_tensor

        if len(self.target_axes) > 1:
            raise ValueError("Slicing multiple axes at once is not yet supported.")

        target_ax = self.target_axes[0]
        base_tensor = self.tensor
        ax_idx = base_tensor.topology.index(target_ax)

        # Build JAX slice tuple
        full_slices = [slice(None)] * len(base_tensor.topology)
        full_slices[ax_idx] = item
        sliced_raw = base_tensor.unwrap()[tuple(full_slices)]

        # 3. Shape Safety & Monad Routing
        if isinstance(item, slice):
            # Range slicing preserves the axis but changes its size
            start, stop, step = item.indices(target_ax.size)
            new_size = len(range(start, stop, step))

            new_ax = _resized_axis(target_ax, new_size)
            new_topology = tuple(new_ax if a == target_ax else a for a in base_tensor.topology)

            chunk_tensor = Tensor(sliced_raw, *new_topology)

            # THE FIX: Wrap it back in a SlicedMonad so you can stitch it later!
            return SlicedMonad(
                base_tensor, target_ax, item, chunk_tensor,
                expected_topology=chunk_tensor.topology, patch_safe=True
            )

        elif isinstance(item, int):
            # Integer indexing completely removes the axis!
            new_topology = tuple(a for a in base_tensor.topology if a != target_ax)
            return Tensor(sliced_raw, *new_topology)

        else:
            raise TypeError("Axiom currently only supports integer, slice, or Tensor indexing.")

class TargetedBundle(NNTargetedBundleStubs):
    """Handles operations targeted across multiple tensors simultaneously."""

    def __init__(self, bundle: 'Bundle', target_axes: Tuple[Axis, ...]):
        self.bundle = bundle
        self.target_axes = target_axes

    def __call__(self, tile_size: int):
        """Create a shared tile plan for every tensor in a parallel Bundle."""
        if len(self.target_axes) != 1:
            raise ValueError("Tile one axis at a time, e.g. (q & k).b(1).s(128).map(fn).")
        from .kernel import tile_axis
        return tile_axis(self.bundle, self.target_axes[0].name, tile_size)

    def __getattr__(self, name: str) -> Any:
        """Universal Dispatcher for Bundles."""
        for tensor in self.bundle.tensors:
            for axis in tensor.topology:
                if axis.name == name:
                    if axis not in self.target_axes:
                        return TargetedBundle(self.bundle, self.target_axes + (axis,))
                    return self

        from . import nn
        if hasattr(nn, name) and callable(getattr(nn, name)) and not name.endswith('_loss'):
            return lambda *args, **kwargs: self.pw(getattr(nn, name), *args, **kwargs)

        func = getattr(jnp, name, None) or getattr(jnn, name, None)
        if callable(func):
            return lambda *args, **kwargs: self.pw(func, *args, **kwargs)

        raise AttributeError(f"Axis, NN function, or JAX primitive '{name}' not found in bundled tensors.")

    def __getitem__(self, key: Any) -> 'Bundle':
        """Parallel patch/slicing across the bundle."""
        if key is None or getattr(key, "_axiom_mesh_axis", False):
            if len(self.target_axes) != 1:
                raise ValueError("Place one logical axis at a time, e.g. (q & k).h[mesh.tp] or .h[None].")
            placed_tensors = []
            for tensor in self.bundle.tensors:
                placed = TargetedTensor(tensor, self.target_axes)[key]
                placed_tensors.append(placed.tensor)
            placed_bundle = Bundle(*placed_tensors)
            target_name = self.target_axes[0].name
            placed_axis = next(axis for axis in placed_tensors[0].topology if axis.name == target_name)
            return TargetedBundle(placed_bundle, (placed_axis,))
        results = [TargetedTensor(t, self.target_axes)[key] for t in self.bundle.tensors]
        return Bundle(*results)

    def proj(self, *target_axes: 'Axis', bias: bool = True, tie: Optional[str] = None, init=None) -> 'Bundle':
        return Bundle(*[TargetedTensor(t, self.target_axes).proj(*target_axes, bias=bias, tie=tie, init=init) for t in
                        self.bundle.tensors])

    def bias(self, init=None, tie: Optional[str] = None) -> 'Bundle':
        return Bundle(*[TargetedTensor(t, self.target_axes).bias(init=init, tie=tie) for t in self.bundle.tensors])

    def gate(self, init=None, tie: Optional[str] = None) -> 'Bundle':
        return Bundle(*[TargetedTensor(t, self.target_axes).gate(init=init, tie=tie) for t in self.bundle.tensors])

    def pw(self, func, *args, tie: Optional[str] = None, **kwargs) -> 'Bundle':
        return Bundle(
            *[
                TargetedTensor(t, self.target_axes).pw(
                    func, *args, tie=tie, **kwargs
                )
                for t in self.bundle.tensors
            ]
        )

    def merge(self, new_axis: Axis) -> 'Bundle':
        """
        Parallel topological flattening across the bundle.
        e.g., (q & k & v).h.w.merge(ax.s)
        """
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.merge(new_axis))

        return Bundle(*results)

    def split(self, *new_axes: Axis) -> 'Bundle':
        """
        Parallel topological splitting across the bundle.
        e.g., (q & k & v).d.split(ax.heads(8), ax.h)
        """
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.split(*new_axes))

        return Bundle(*results)

    def unfold(self, window_axis: 'Axis', step: int = 1) -> 'Bundle':
        return Bundle(
            *[TargetedTensor(t, self.target_axes).unfold(window_axis, step=step) for t in self.bundle.tensors])

    def scan(self, func, init=None, associative: bool = False) -> Any:
        import jax
        import jax.numpy as jnp
        from .compiler import compiler_state  # <-- Required for Ghost Pass!

        scan_ax = self.target_axes[0]
        raw_elems, inv_perms, inner_topologies = [], [], []

        for tensor in self.bundle.tensors:
            idx = tensor.topology.index(scan_ax)
            perm = [idx] + [i for i in range(len(tensor.topology)) if i != idx]
            inv_perms.append([perm.index(i) for i in range(len(tensor.topology))])
            raw_elems.append(jnp.transpose(tensor.unwrap(), perm))
            inner_topologies.append(tuple(a for a in tensor.topology if a != scan_ax))

        if associative:
            def wrapped_assoc(raw_left, raw_right):
                left_tensors, right_tensors, chunk_ax = [], [], None
                for r_l, r_r, top in zip(raw_left, raw_right, inner_topologies):
                    current_top = top
                    if hasattr(r_l, 'shape') and len(r_l.shape) == len(top) + 1:
                        if chunk_ax is None: chunk_ax = Axis("_chunk", r_l.shape[0])
                        current_top = (chunk_ax,) + top
                    left_tensors.append(Tensor(r_l, *current_top))
                    right_tensors.append(Tensor(r_r, *current_top))

                out_tuple = func(tuple(left_tensors), tuple(right_tensors))
                return tuple(out.unwrap() for out in out_tuple)

            out_raw_elems = jax.lax.associative_scan(wrapped_assoc, tuple(raw_elems))
            return tuple(Tensor(jnp.transpose(raw_out, inv_perm), *t.topology) for raw_out, inv_perm, t in
                         zip(out_raw_elems, inv_perms, self.bundle.tensors))

        else:
            if init is None:
                raise ValueError("Sequential scan requires 'init' state(s). Or use associative=True.")

            init_tensors = init.tensors if hasattr(init, 'tensors') else init
            init_raws = tuple(t.unwrap() for t in init_tensors)
            init_tops = tuple(t.topology for t in init_tensors)

            # --- 1. Weight Tying Scope Setup ---
            scan_scope = f"scan_bundle_{compiler_state.param_counter}"
            prev_override = compiler_state.tied_scope_override
            start_counter = compiler_state.param_counter

            # --- 2. The Ghost Pass Bypass ---
            if compiler_state.is_initializing:
                compiler_state.tied_scope_override = scan_scope

                # Extract t=0 for all elements (raw_elems is already transposed, so axis 0 is seq!)
                mock_xt_tensors = [Tensor(jnp.take(r, 0, axis=0), *top) for r, top in zip(raw_elems, inner_topologies)]

                # Eagerly execute to allocate physical arrays!
                _ = func(Bundle(*init_tensors), Bundle(*mock_xt_tensors))

            # --- 3. The Trace Pass ---
            compiler_state.tied_scope_override = scan_scope

            def wrapped_func(raw_carry_tuple, raw_xt_tuple):
                # Lock weights for recurrent execution!
                compiler_state.param_counter = start_counter

                carry_tensors = tuple(Tensor(r, *top) for r, top in zip(raw_carry_tuple, init_tops))
                xt_tensors = tuple(Tensor(r, *top) for r, top in zip(raw_xt_tuple, inner_topologies))

                new_carry, out_y = func(Bundle(*carry_tensors), Bundle(*xt_tensors))

                new_carry_tensors = new_carry.tensors if hasattr(new_carry, 'tensors') else new_carry
                out_y_tensors = out_y.tensors if hasattr(out_y, 'tensors') else out_y

                return tuple(t.unwrap() for t in new_carry_tensors), tuple(t.unwrap() for t in out_y_tensors)

            final_carry_raw, y_seq_raw = jax.lax.scan(wrapped_func, init_raws, tuple(raw_elems))

            # --- 4. Cleanup & State Restoration ---
            compiler_state.tied_scope_override = prev_override
            allocated = len([k for k in compiler_state.params if k.startswith(scan_scope)])
            compiler_state.param_counter = start_counter + allocated

            final_carry = Bundle(*[Tensor(r, *top) for r, top in zip(final_carry_raw, init_tops)])
            y_seq = Bundle(*[Tensor(jnp.transpose(r, inv_perm), *t.topology) for r, inv_perm, t in
                             zip(y_seq_raw, inv_perms, self.bundle.tensors)])

            return final_carry, y_seq

    def rename(self, *new_axes: 'Axis') -> 'Bundle':
        if len(new_axes) == 1: new_axes = new_axes * len(self.bundle.tensors)
        return Bundle(*[TargetedTensor(t, self.target_axes).rename(new_ax) for t, new_ax in zip(self.bundle.tensors, new_axes)])

    def join(self) -> 'TargetedTensor':
        import jax.numpy as jnp

        target_ax = self.target_axes[0]
        base_top = self.bundle.tensors[0].topology
        for tensor in self.bundle.tensors[1:]:
            if tuple(axis.name for axis in tensor.topology) != tuple(axis.name for axis in base_top):
                raise ValueError("Cannot join tensors with different logical axis names/order.")
            merged = []
            for left, right in zip(base_top, tensor.topology):
                # The joined axis intentionally has a different extent in
                # each bundle member; only its placement must agree.
                if left.name == target_ax.name:
                    merged.append(_merged_axis(_resized_axis(left, None), _resized_axis(right, None)))
                else:
                    merged.append(_merged_axis(left, right))
            base_top = tuple(merged)
        target_ax = base_top[base_top.index(target_ax)]

        raw_arrays = []
        for t in self.bundle.tensors:
            if t.topology != base_top:
                try:
                    perm = [t.topology.index(a) for a in base_top]
                    raw_arrays.append(jnp.transpose(t.unwrap(), axes=perm))
                except ValueError:
                    raise ValueError(
                        f"Cannot join tensors with incompatible topologies. "
                        f"Expected axes {base_top}, but got {t.topology}"
                    )
            else:
                raw_arrays.append(t.unwrap())

        ax_idx = base_top.index(target_ax)

        # Calculate the new total size along the joined axis
        total_size = sum(t.topology[t.topology.index(target_ax)].size for t in self.bundle.tensors)

        # Build the new topology using base_top and retain the target layout.
        new_topology = list(base_top)
        new_topology[ax_idx] = _resized_axis(target_ax, total_size)

        return Tensor(jnp.concatenate(raw_arrays, axis=ax_idx), *new_topology)

    def pad(self, *pad_widths: Tuple[int, int], fill: float = 0.0) -> 'Bundle':
        return Bundle(*[TargetedTensor(t, self.target_axes).pad(*pad_widths, fill=fill) for t in self.bundle.tensors])

    def mask(self, func, fill: float = 0.0) -> 'Bundle':
        return Bundle(*[TargetedTensor(t, self.target_axes).mask(func, fill=fill) for t in self.bundle.tensors])

    def vmask(self, func, fill: float = 0.0) -> 'Bundle':
        bool_mask = func(*[t.unwrap() for t in self.bundle.tensors])
        return Bundle(*[Tensor(jnp.where(bool_mask, fill, t.unwrap()), *t.topology) for t in self.bundle.tensors])

    def __iter__(self):
        """Allows Pythonic unpacking across a bundle: (q & k).heads"""
        if len(self.target_axes) > 1:
            raise ValueError("Can only unpack a single targeted axis at a time.")

        target_ax = self.target_axes[0]

        if target_ax.size is None:
            raise ValueError(f"Cannot unpack axis '{target_ax.name}' because its size is unknown.")

        for i in range(target_ax.size):
            yield self[i]


class Bundle(NNBundleStubs):
    """Wraps multiple Tensors to perform parallel, fused operations."""

    def __init__(self, *tensors: Tensor):
        self.tensors = tensors

    def __and__(self, other: 'Tensor') -> 'Bundle':
        return Bundle(*self.tensors, other)

    def __iter__(self):
        return iter(self.tensors)

    def __getattr__(self, name: str) -> Any:
        # 1. Target an Axis
        for tensor in self.tensors:
            for axis in tensor.topology:
                if axis.name == name:
                    return TargetedBundle(self, (axis,))

        # 2. Universal Dispatcher: Pure Pointwise mapping across the Bundle
        from . import nn
        import jax.numpy as jnp
        import jax.nn as jnn
        import inspect

        func = getattr(nn, name, None)
        if func and callable(func) and not name.endswith('_loss'):
            pass
        else:
            func = getattr(jnp, name, None) or getattr(jnn, name, None)

        if callable(func):
            requires_axis = False
            try:
                if 'axis' in inspect.signature(func).parameters: requires_axis = True
            except ValueError:
                pass

            if requires_axis or getattr(func, '_is_axiom_nn', False):
                raise ValueError(f"Mathematical Ambiguity: '{name}' requires a target axis. Use (x & y).axis.{name}()")

            def bound_pointwise(*args, **kwargs):
                return Bundle(*[getattr(t, name)(*args, **kwargs) for t in self.tensors])

            return bound_pointwise

        raise AttributeError(f"Axis '{name}' not found in bundled tensors.")

    def _pairwise(self, op_name: str) -> 'Tensor':
        out = self.tensors[0]
        for tensor in self.tensors[1:]: out = getattr(out, op_name)(tensor)
        return out

    def minimum(self) -> 'Tensor':
        return self._pairwise("minimum")

    def maximum(self) -> 'Tensor':
        return self._pairwise("maximum")

    def min(self) -> 'Tensor':
        return self.minimum()

    def max(self) -> 'Tensor':
        return self.maximum()

    def mean(self) -> 'Bundle':
        return Bundle(*[t.mean() for t in self.tensors])

    def sum(self) -> 'Bundle':
        return Bundle(*[t.sum() for t in self.tensors])

    def repeat(self, func: callable, times: int) -> 'Bundle':
        """Executes a function multiple times over a bundle using tied parameters."""
        import jax.lax as lax

        repeat_scope = f"repeat_block_{compiler_state.param_counter}"
        prev_override = compiler_state.tied_scope_override
        start_counter = compiler_state.param_counter

        prev_init = compiler_state.is_initializing
        end_counter = start_counter

        try:
            # 1. The Ghost Pass
            compiler_state.tied_scope_override = repeat_scope

            # THE FIX: Bypass jax.checkpoint tapes!
            compiler_state.is_initializing = True
            _ = func(self)

            end_counter = compiler_state.param_counter
            compiler_state.is_initializing = prev_init

            # 2. The Trace Pass
            def _scan_body(carry_raw_tuple, _):
                compiler_state.param_counter = start_counter

                carry_tensors = [Tensor(raw, *orig_t.topology) for raw, orig_t in zip(carry_raw_tuple, self.tensors)]
                out_bundle = func(Bundle(*carry_tensors))

                out_raw_tuple = []
                for in_t, out_t, raw_in in zip(self.tensors, out_bundle.tensors, carry_raw_tuple):
                    if in_t.topology != out_t.topology:
                        raise ValueError("repeat requires matched topologies for XLA.")
                    out_raw_tuple.append(out_t.unwrap().astype(raw_in.dtype))

                return tuple(out_raw_tuple), None

            final_raw_tuple, _ = lax.scan(_scan_body, tuple(t.unwrap() for t in self.tensors), None, length=times)
            return Bundle(*[Tensor(raw, *orig_t.topology) for raw, orig_t in zip(final_raw_tuple, self.tensors)])

        finally:
            # 3. Cleanup
            compiler_state.tied_scope_override = prev_override
            compiler_state.param_counter = end_counter
            compiler_state.is_initializing = prev_init

    def astype(self, dtype) -> 'Bundle':
        """Safely casts an entire bundle of tensors to a new precision."""
        return Bundle(*[t.astype(dtype) for t in self.tensors])


class Tensor(NNTensorStubs):
    """The core Axiom Tensor wrapper enforcing named-axis topologies."""

    def __init__(self, raw_tensor: Any, *axes: Axis):
        import jax.numpy as jnp

        if not hasattr(raw_tensor, "shape") or not hasattr(raw_tensor, "dtype"):
            raw_tensor = jnp.asarray(raw_tensor)

        self._tensor = raw_tensor
        self._axes = axes
        self._validate_topology()

    def _validate_topology(self):
        seen_names = set()
        seen_mesh_dimensions = set()
        mesh_ids = set()
        for axis in self._axes:
            if axis.name in seen_names:
                raise ValueError(
                    f"Topological Ambiguity: Tensor contains duplicate axis '{axis.name}'.\n"
                    f"Axiom enforces Strict Identity. If these represent distinct vector spaces, "
                    f"you must use .rename() to differentiate them (e.g., '{axis.name}_1', '{axis.name}_2')."
                )
            seen_names.add(axis.name)
            if axis.placement is not None:
                mesh_ids.add(axis.placement.mesh_id)
                if axis.placement.name in seen_mesh_dimensions:
                    raise ValueError(
                        f"Mesh axis '{axis.placement.name}' is assigned to more than one tensor dimension. "
                        "Split/merge logical axes so every physical dimension has a unique mesh axis."
                    )
                seen_mesh_dimensions.add(axis.placement.name)
        if len(mesh_ids) > 1:
            raise ValueError("A Tensor cannot combine axes from different ax.mesh() instances.")

        if hasattr(self._tensor, 'shape'):
            if len(self._tensor.shape) != len(self._axes):
                raise ValueError(
                    f"Topology mismatch: Array has {len(self._tensor.shape)} dims, given {len(self._axes)} axes.")
            for dim_size, axis in zip(self._tensor.shape, self._axes):
                if axis.size is not None and axis.size != dim_size:
                    raise ValueError(f"Size mismatch on '{axis.name}': expected {axis.size}, got {dim_size}.")

    def unwrap(self) -> Any:
        return self._tensor

    @property
    def topology(self) -> Tuple[Axis, ...]:
        return self._axes

    @property
    def debugger_stats(self):
        """
        Evaluated on-demand by the PyCharm debugger dropdown.
        Provides safe, beautifully formatted numerical observability.
        """
        import jax
        import jax.numpy as jnp
        import math

        raw_val = self.unwrap() if hasattr(self, 'unwrap') else None

        if isinstance(raw_val, jax.core.Tracer):
            return "Currently Traced (No concrete values)"

        try:
            def fmt(v):
                v = float(v)
                if math.isnan(v): return "NaN ⚠️"
                if math.isinf(v): return "Inf ⚠️"
                if v == 0: return "0.0000"
                # Use scientific notation for very small or very large numbers
                if abs(v) < 1e-4 or abs(v) > 1e4:
                    return f"{v:.4e}"
                # Standard 4-decimal format for everything else
                return f"{v:.4f}"

            return {
                "mean": fmt(jnp.mean(raw_val)),
                "var": fmt(jnp.var(raw_val)),
                "std": fmt(jnp.std(raw_val)),
                "min": fmt(jnp.min(raw_val)),
                "max": fmt(jnp.max(raw_val))
            }
        except Exception:
            return "Stats unavailable (Empty or unsupported dtype)"

    def param(self, name: Optional[str] = None, tie: Optional[Tie] = None) -> 'Tensor':
        """Registers the tensor as a trainable parameter for the AOT compiler."""
        self._is_param = True

        explicit_tie = tie if isinstance(tie, str) else (tie.name if tie else None)
        fallback = name if name else "param"

        true_name = compiler_state.get_scoped_name(explicit_name=explicit_tie, fallback_prefix=fallback)
        self._param_name = true_name

        import jax
        is_tracing = isinstance(self.unwrap(), jax.core.Tracer)

        # THE FIX: Absolute priority to is_tracing! Never allocate Tracers.
        if is_tracing:
            if true_name not in compiler_state.params:
                raise RuntimeError(
                    f"Axiom Tracer Leak Prevention: Parameter '{true_name}' was requested during JAX tracing, "
                    f"but was not allocated eagerly! You must initialize your model with a dummy forward pass "
                    f"BEFORE compiling it."
                )
        else:
            # Eager execution: Safe to allocate memory
            if true_name not in compiler_state.params:
                if compiler_state.strict_params:
                    raise RuntimeError(
                        f"Axiom parameter '{true_name}' is missing from an initialized model/exported parameter "
                        "dictionary. Reinitialize the model or pass the complete parameter tree."
                    )
                compiler_state.params[true_name] = self.unwrap()

        from .layout import ParameterLayout
        layout = ParameterLayout(tuple(self.topology))
        existing_layout = compiler_state.param_layouts.get(true_name)
        if existing_layout is None:
            compiler_state.param_layouts[true_name] = layout
        elif existing_layout.metadata() != layout.metadata():
            raise ValueError(
                f"Tied parameter '{true_name}' was requested with incompatible axis placement metadata."
            )

        return Tensor(compiler_state.params[true_name], *self.topology)

    def pw(self, func, **kwargs) -> 'Tensor':
        return Tensor(func(self.unwrap(), **kwargs), *self._axes)

    def minimum(self, other) -> 'Tensor':
        return self._broadcast_op(other, jnp.minimum)

    def maximum(self, other) -> 'Tensor':
        return self._broadcast_op(other, jnp.maximum)

    def mean(self) -> 'Tensor':
        import jax.numpy as jnp
        return Tensor(jnp.mean(self.unwrap()))

    def sum(self) -> 'Tensor':
        import jax.numpy as jnp
        return Tensor(jnp.sum(self.unwrap()))

    def broadcast(self, *new_axes: 'Axis') -> 'Tensor':
        """
        Projects the tensor into new geometric dimensions without allocating dummy variables.
        """
        import jax.numpy as jnp

        # 1. The Hardened Geometric Shield
        current_names = {a.name for a in self.topology}
        for a in new_axes:
            if a.name in current_names:
                raise ValueError(
                    f"Topological Ambiguity: Cannot broadcast to '{a.name}'. "
                    f"Axis already exists in the current topology."
                )

            # THE FIX: Explicit integer and bounds checking
            if not isinstance(getattr(a, 'size', None), int) or a.size <= 0:
                raise ValueError(
                    f"Unknown Geometry: Broadcast target axis '{getattr(a, 'name', '?')}' "
                    "must have a strictly defined size."
                )

        # 2. Expand the physical dimensions
        raw_val = self.unwrap()
        for _ in new_axes:
            raw_val = jnp.expand_dims(raw_val, axis=-1)

        # 3. Calculate the target geometry
        final_topology = tuple(self.topology) + new_axes
        target_shape = tuple(a.size for a in final_topology)

        # 4. Zero-Cost XLA Broadcast
        broadcasted_raw = jnp.broadcast_to(raw_val, target_shape)

        return Tensor(broadcasted_raw, *final_topology)

    def repeat(self, func: callable, times: int) -> 'Tensor':
        """Executes a function multiple times using the exact same tied parameters."""
        import jax.lax as lax

        repeat_scope = f"repeat_block_{compiler_state.param_counter}"
        prev_override = compiler_state.tied_scope_override
        start_counter = compiler_state.param_counter

        # Capture the original state
        prev_init = compiler_state.is_initializing
        end_counter = start_counter

        try:
            # 1. The Ghost Pass
            compiler_state.tied_scope_override = repeat_scope

            # THE FIX: Temporarily force initialization to bypass jax.checkpoint tapes!
            compiler_state.is_initializing = True
            _ = func(self)

            end_counter = compiler_state.param_counter

            # Restore the true tracing state before the actual scan loop
            compiler_state.is_initializing = prev_init

            # 2. The Trace Pass
            def _scan_body(carry_raw, _):
                compiler_state.param_counter = start_counter
                out_tensor = func(Tensor(carry_raw, *self.topology))
                out_raw = out_tensor.unwrap().astype(carry_raw.dtype)
                return out_raw, None

            final_raw, _ = lax.scan(_scan_body, self.unwrap(), None, length=times)
            return Tensor(final_raw, *self.topology)

        finally:
            # 3. Cleanup
            compiler_state.tied_scope_override = prev_override
            compiler_state.param_counter = end_counter
            compiler_state.is_initializing = prev_init  # Failsafe restore

    def vmask(self, func, fill: float = 0.0) -> 'Tensor':
        """Value-based masking. Masks the tensor based on its own values."""
        import jax.numpy as jnp
        bool_mask = func(self.unwrap())
        result_raw = jnp.where(bool_mask, fill, self.unwrap())
        return Tensor(result_raw, *self.topology)

    def _get_union_topology(self, other: 'Tensor') -> Tuple[Axis, ...]:
        union = list(self._axes)
        for ax_other in other.topology:
            if ax_other not in union:
                union.append(ax_other)
            else:
                index = union.index(ax_other)
                union[index] = _merged_axis(union[index], ax_other)
        return tuple(union)

    def _align_to(self, target_axes: Tuple[Axis, ...]) -> Any:
        current_indices = [self._axes.index(ta) for ta in target_axes if ta in self._axes]
        transposed = jnp.transpose(self.unwrap(), current_indices)
        final_shape = [ta.size if ta in self._axes else 1 for ta in target_axes]
        return jnp.reshape(transposed, final_shape)

    def _broadcast_op(self, other, op_func) -> 'Tensor':
        if hasattr(other, 'chunk_tensor'):
            other = other.chunk_tensor
        elif hasattr(other, 'tensor') and hasattr(other, 'target_axes'):
            other = other.tensor

        if not isinstance(other, Tensor):
            return Tensor(op_func(self.unwrap(), other), *self._axes)

        union_axes = self._get_union_topology(other)
        return Tensor(op_func(self._align_to(union_axes), other._align_to(union_axes)), *union_axes)

    def astype(self, dtype) -> 'Tensor':
        """Safely casts the underlying JAX array to a new precision."""
        return Tensor(self.unwrap().astype(dtype), *self.topology)

    def stop_grad(self) -> 'Tensor':
        return self.pw(jax.lax.stop_gradient)

    def __add__(self, other):
        return self._broadcast_op(other, operator.add)

    def __radd__(self, other):
        return self._broadcast_op(other, operator.add)

    def __sub__(self, other):
        return self._broadcast_op(other, operator.sub)

    def __rsub__(self, other):
        return other._broadcast_op(self, operator.sub) if isinstance(other, Tensor) else Tensor(
            operator.sub(other, self.unwrap()), *self._axes)

    def __mul__(self, other):
        return self._broadcast_op(other, operator.mul)

    def __rmul__(self, other):
        return self._broadcast_op(other, operator.mul)

    def __truediv__(self, other):
        return self._broadcast_op(other, operator.truediv)

    def __rtruediv__(self, other):
        return other._broadcast_op(self, operator.truediv) if isinstance(other, Tensor) else Tensor(
            operator.truediv(other, self.unwrap()), *self._axes)

    def __pow__(self, other):
        return self._broadcast_op(other, operator.pow)

    def __rpow__(self, other):
        return other._broadcast_op(self, operator.pow) if isinstance(other, Tensor) else Tensor(
            operator.pow(other, self.unwrap()), *self._axes)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        shared_axes = [a for a in self.topology if a in other.topology]
        return TargetedTensor(self, (shared_axes[-1],)) @ other

    def __getattr__(self, name: str) -> Any:
        """Universal Dispatcher for Pointwise Tensor calls."""
        for axis in self._axes:
            if axis.name == name: return TargetedTensor(self, (axis,))

        from . import nn
        func = getattr(nn, name, None)
        if func and callable(func) and not name.endswith('_loss'):
            pass
        else:
            func = getattr(jnp, name, None) or getattr(jnn, name, None)

        if callable(func):
            requires_axis = False
            try:
                if 'axis' in inspect.signature(func).parameters: requires_axis = True
            except ValueError:
                pass

            if requires_axis or getattr(func, '_is_axiom_nn', False):
                raise ValueError(
                    f"Mathematical Ambiguity: '{name}' requires a target axis. You must target an axis first! (e.g., x.d.{name}())")

            def bound_pointwise(*args, **kwargs):
                return Tensor(func(self.unwrap(), *args, **kwargs), *self.topology)

            return bound_pointwise

        raise AttributeError(
            f"Tensor has no axis, NN function, or JAX primitive '{name}'. Topology: {[a.name for a in self._axes]}")

    def __class_getitem__(cls, item: Any) -> Type:
        return cls

    def __neg__(self) -> 'Tensor':
        return Tensor(-self.unwrap(), *self._axes)

    def __and__(self, other: 'Tensor') -> 'Bundle':
        return Bundle(self, other)

    def item(self) -> float:
        import numpy as np
        return float(np.array(self.unwrap()).item())

    def __float__(self) -> float:
        return self.item()

    def __str__(self):
        """Controls what happens when the user calls print(tensor)"""
        return str(self.unwrap())

    def __repr__(self):
        """
        Keeps PyCharm inline hints completely pristine.
        Example: Tensor[b[4], s[32], d[16]]
        """
        topo_strings = [f"{getattr(a, 'name', '?')}[{a.size if a.size is not None else '?'}]" for a in self.topology]
        return f"Tensor[{', '.join(topo_strings)}]"

    def __format__(self, format_spec: str) -> str:
        """Safely formats the tensor. Falls back to __str__ if not formatting a scalar."""
        if format_spec == "":
            return str(self)
        return format(self.item(), format_spec)


def wrap(raw_tensor: Any, *axes: Axis) -> Tensor:
    return Tensor(raw_tensor, *axes)


class RoutedContext:
    def __init__(self, tensor: Tensor, target_axes: Tuple[Axis, ...], indices: Tensor):
        self.tensor = tensor
        self.target_ax = target_axes[0]
        self.indices = indices

    def gather(self) -> Tensor:
        ax_idx = self.tensor.topology.index(self.target_ax)
        safe_indices = self.indices.unwrap().astype(jnp.int32)
        res_raw = jnp.take(self.tensor.unwrap(), safe_indices, axis=ax_idx)
        new_top = list(self.tensor.topology)
        new_top = new_top[:ax_idx] + list(self.indices.topology) + new_top[ax_idx + 1:]
        return Tensor(res_raw, *new_top)


def decay_monads(x):
    if hasattr(x, "chunk_tensor"): return x.chunk_tensor
    if isinstance(x, Bundle): return Bundle(*[decay_monads(t) for t in x.tensors])
    if isinstance(x, tuple): return tuple(decay_monads(v) for v in x)
    if isinstance(x, list): return [decay_monads(v) for v in x]
    if isinstance(x, dict): return {k: decay_monads(v) for k, v in x.items()}
    return x


def _flatten_tensor(tensor):
    # Axis equality is intentionally name-based for named contraction.  The
    # PyTree auxiliary data must additionally include placement, otherwise JAX
    # could incorrectly reuse a compiled executable for ``d[dp]`` and
    # ``d[tp]`` tensors with the same shape.
    axis_data = tuple((axis.name, axis.size, axis.placement, axis.replicated) for axis in tensor.topology)
    return (tensor.unwrap(),), axis_data


def _unflatten_tensor(axis_data, children):
    return Tensor(children[0], *(Axis(name, size, placement, replicated=replicated)
                                 for name, size, placement, replicated in axis_data))


jax.tree_util.register_pytree_node(Tensor, _flatten_tensor, _unflatten_tensor)
