from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type

if TYPE_CHECKING:
    # PyCharm sees this and loads the autocomplete hints!
    from ._nn_stubs import NNTensorStubs, NNTargetedTensorStubs, NNTargetedBundleStubs
else:
    # At runtime, Python sees these as empty bases so they don't break __getattr__!
    class NNTensorStubs: pass
    class NNTargetedTensorStubs: pass
    class NNTargetedBundleStubs: pass


class CompilerState:
    """Global tracker for AOT parameter allocation."""
    def __init__(self):
        self.is_initializing = False
        self.params = {}
        self.param_counter = 0

compiler_state = CompilerState()

class Tie:
    def __init__(self, name: str):
        self.name = name


class Axis:
    """Represents a logical dimension in Axiom."""

    def __init__(self, name: str, size: Optional[int] = None):
        self.name = name
        self.size = size
        self.mesh_dim: Optional[str] = None

    def __call__(self, size: Any) -> 'Axis':
        new_ax = Axis(self.name, int(size))
        new_ax.mesh_dim = self.mesh_dim
        return new_ax

    def shard(self, mesh_dim: str) -> 'Axis':
        self.mesh_dim = mesh_dim
        return self

    def __floordiv__(self, other: int) -> int:
        if self.size is None:
            raise ValueError(f"Cannot divide template axis '{self.name}' because it has no size. Use a targeted tensor (e.g., x.{self.name} // {other}) instead.")
        return self.size // other

    def __mul__(self, other: int) -> int:
        if self.size is None:
            raise ValueError(f"Cannot multiply template axis '{self.name}' because it has no size. Use a targeted tensor (e.g., x.{self.name} * {other}) instead.")
        return self.size * other

    def __add__(self, other: int) -> int:
        if self.size is None:
            raise ValueError(f"Cannot add to template axis '{self.name}' because it has no size. Use a targeted tensor (e.g., x.{self.name} + {other}) instead.")
        return self.size + other

    def __repr__(self):
        size_str = f"={self.size}" if self.size is not None else ""
        return f"Axis('{self.name}'{size_str})"

    def __hash__(self): return hash(self.name)

    def __eq__(self, other): return isinstance(other, Axis) and self.name == other.name


class _AxisNamespace:
    def __call__(self, name: str, size: Optional[int] = None) -> Axis:
        return Axis(name, size)

    # Compiler namespace aliases
    def model(self, fn) -> 'AxiomModel':
        """Allows `model = ax.model(fn)` or `@ax.model` decorator."""
        from .compiler import AxiomModel
        return AxiomModel(fn)

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

    def __getattr__(self, name: str) -> Axis:
        """
        The Magic Inline Creator!
        When you type `ax.b`, it dynamically creates an Axis internally named "b".
        You can then call it like `ax.b(32)` to set its size!
        """
        return Axis(name)


ax = _AxisNamespace()


class Mesh:
    """Hardware topology definition for distributed XLA sharding."""
    def __init__(self, devices, axis_names):
        self.devices = devices
        self.axis_names = axis_names


class Tie:
    def __init__(self, name: str):
        self.name = name


class SlicedMonad:
    """
    Lazy slice / optional patch transaction.

    Semantics:
      - If consumed as a value, it behaves like its chunk_tensor.
      - If closed with [:], it commits the chunk back into the original tensor.
      - Patch commits are only allowed for patch-safe, topology-preserving transforms.
    """

    def __init__(
        self,
        original_tensor,
        target_ax,
        slice_obj,
        chunk_tensor,
        expected_topology=None,
        patch_safe: bool = True,
        unsafe_reason: Optional[str] = None,
    ):
        self.original_tensor = original_tensor
        self.target_ax = target_ax
        self.slice_obj = slice_obj
        self.chunk_tensor = chunk_tensor
        self.expected_topology = tuple(expected_topology or chunk_tensor.topology)
        self.patch_safe = patch_safe
        self.unsafe_reason = unsafe_reason

    # -------------------------
    # Tensor-like value decay
    # -------------------------

    def _as_tensor(self):
        return self.chunk_tensor

    def unwrap(self):
        return self.chunk_tensor.unwrap()

    @property
    def topology(self):
        return self.chunk_tensor.topology

    def _topology_signature(self, topology):
        # Axis.__eq__ currently compares only by name, so strict commit checks
        # must compare both name and size explicitly.
        return tuple((a.name, a.size) for a in topology)

    def _current_target_ax(self):
        """
        Finds the current chunk axis corresponding to the original patched axis.
        This avoids using a stale Axis object from the parent tensor.
        """
        for a in self.chunk_tensor.topology:
            if a.name == self.target_ax.name:
                return a

        raise ValueError(
            f"Cannot target original patch axis '{self.target_ax.name}' because "
            f"it is no longer present in the sliced chunk topology: "
            f"{[repr(a) for a in self.chunk_tensor.topology]}"
        )

    def _wrap(
        self,
        result,
        *,
        patch_safe: Optional[bool] = None,
        unsafe_reason: Optional[str] = None,
    ):
        """
        Re-wrap Tensor results into the same pending patch transaction.
        Non-Tensor results are returned as-is.
        """
        if hasattr(result, "chunk_tensor"):
            result = result.chunk_tensor

        if not isinstance(result, Tensor):
            return result

        if patch_safe is None:
            patch_safe = self.patch_safe

        if unsafe_reason is None:
            unsafe_reason = self.unsafe_reason

        return SlicedMonad(
            self.original_tensor,
            self.target_ax,
            self.slice_obj,
            result,
            expected_topology=self.expected_topology,
            patch_safe=patch_safe,
            unsafe_reason=unsafe_reason,
        )

    def _unsafe(self, reason: str):
        return SlicedMonad(
            self.original_tensor,
            self.target_ax,
            self.slice_obj,
            self.chunk_tensor,
            expected_topology=self.expected_topology,
            patch_safe=False,
            unsafe_reason=reason,
        )

    # -------------------------
    # Common direct ops
    # -------------------------

    def pw(self, func, **kwargs) -> 'SlicedMonad':
        target = self._current_target_ax()
        new_chunk = TargetedTensor(self.chunk_tensor, (target,)).pw(func, **kwargs)
        return self._wrap(new_chunk)

    def proj(self, *target_axes, bias: bool = False, tie: Optional[str] = None, init=None) -> 'SlicedMonad':
        """
        Patch-safe only for proj() with no explicit target axes.

        Legal:
            x.d[half:].proj()[:]

        Illegal as patch:
            x.d[half:].proj(ax.d2(128))[:]
            x.d[half:].proj(ax.d(128))[:]

        Explicit projection is still usable as a value; it just cannot be committed
        through the monad.
        """
        target = self._current_target_ax()

        explicit_projection = len(target_axes) > 0
        new_chunk = TargetedTensor(self.chunk_tensor, (target,)).proj(
            *target_axes,
            bias=bias,
            tie=tie,
            init=init,
        )

        if explicit_projection:
            return self._wrap(
                new_chunk,
                patch_safe=False,
                unsafe_reason=(
                    "Explicit proj(...) inside a sliced patch is not commit-safe. "
                    "Use proj() with no explicit axes for same-axis patch projection, "
                    "or stitch manually."
                ),
            )

        return self._wrap(new_chunk)

    def bias(self, init=None, tie: Optional[str] = None) -> 'SlicedMonad':
        target = self._current_target_ax()
        new_chunk = TargetedTensor(self.chunk_tensor, (target,)).bias(init=init, tie=tie)
        return self._wrap(new_chunk)

    def gate(self, init=None, tie: Optional[str] = None) -> 'SlicedMonad':
        target = self._current_target_ax()
        new_chunk = TargetedTensor(self.chunk_tensor, (target,)).gate(init=init, tie=tie)
        return self._wrap(new_chunk)

    # -------------------------
    # Attribute / axis routing
    # -------------------------

    def __getattr__(self, name):
        # Axis targeting should preserve patch context.
        for axis in self.chunk_tensor.topology:
            if axis.name == name:
                return TargetedSlicedMonad(self, (axis,))

        attr = getattr(self.chunk_tensor, name)

        if callable(attr):
            def wrapped(*args, **kwargs):
                return self._wrap(attr(*args, **kwargs))
            return wrapped

        return attr

    # -------------------------
    # Arithmetic
    # -------------------------

    def _is_monad_like(self, x):
        return hasattr(x, "chunk_tensor")

    def _unwrap_other(self, other):
        if hasattr(other, "chunk_tensor"):
            return other.chunk_tensor
        if hasattr(other, "tensor") and hasattr(other, "target_axes"):
            return other.tensor
        return other

    def _binary_value_or_patch(self, other, op):
        """
        Arithmetic policy:

          - monad op monad:
              dissolve to a pure Tensor, because there are now two competing
              patch origins and no unambiguous commit target.

          - monad op scalar/plain Tensor:
              preserve the patch context, because this is a simple transform
              of one sliced region.
        """
        other_is_monad = self._is_monad_like(other)
        other_unwrapped = self._unwrap_other(other)
        result = op(self.chunk_tensor, other_unwrapped)

        if other_is_monad:
            return result

        return self._wrap(result)

    def _rbinary_value_or_patch(self, other, op):
        other_is_monad = self._is_monad_like(other)
        other_unwrapped = self._unwrap_other(other)
        result = op(other_unwrapped, self.chunk_tensor)

        if other_is_monad:
            return result

        return self._wrap(result)

    def __add__(self, other):
        import operator
        return self._binary_value_or_patch(other, operator.add)

    def __radd__(self, other):
        import operator
        return self._rbinary_value_or_patch(other, operator.add)

    def __sub__(self, other):
        import operator
        return self._binary_value_or_patch(other, operator.sub)

    def __rsub__(self, other):
        import operator
        return self._rbinary_value_or_patch(other, operator.sub)

    def __mul__(self, other):
        import operator
        return self._binary_value_or_patch(other, operator.mul)

    def __rmul__(self, other):
        import operator
        return self._rbinary_value_or_patch(other, operator.mul)

    def __truediv__(self, other):
        import operator
        return self._binary_value_or_patch(other, operator.truediv)

    def __rtruediv__(self, other):
        import operator
        return self._rbinary_value_or_patch(other, operator.truediv)

    def __neg__(self):
        return self._wrap(-self.chunk_tensor)

    def __matmul__(self, other):
        # Matmul is value-style by default. It should not preserve patch context.
        return self.chunk_tensor @ self._unwrap_other(other)

    def __and__(self, other):
        # Bundling is value-style by default. It should not preserve patch context.
        return self.chunk_tensor & self._unwrap_other(other)

    # -------------------------
    # Commit
    # -------------------------

    def __getitem__(self, key) -> 'Tensor':
        """
        The Unslice Closer [:].

        Commits only if:
          1. the operation was patch-safe
          2. the transformed chunk has exactly the original sliced topology
        """
        if key != slice(None):
            raise ValueError("Use [:] to stitch the sliced monad back into the parent tensor.")

        import jax.numpy as jnp

        if not self.patch_safe:
            raise ValueError(
                "Cannot commit sliced patch because this slice went through an unsafe operation. "
                f"{self.unsafe_reason or ''}"
            )

        expected_sig = self._topology_signature(self.expected_topology)
        actual_sig = self._topology_signature(self.chunk_tensor.topology)

        if actual_sig != expected_sig:
            raise ValueError(
                "Cannot commit sliced patch because the chunk topology changed. "
                f"Expected {expected_sig}, got {actual_sig}. "
                "Sliced patch commits only support topology-preserving transforms. "
                "For projections, reshapes, renames, unfolds, reductions, or other topology-changing "
                "operations, stitch manually."
            )

        orig_raw = self.original_tensor.unwrap()
        chunk_raw = self.chunk_tensor.unwrap()
        ax_idx = self.original_tensor.topology.index(self.target_ax)

        # This monad currently supports simple static Python slices.
        if not isinstance(self.slice_obj, slice):
            raise ValueError(
                "Sliced patch commit currently only supports Python slice objects. "
                f"Got {type(self.slice_obj)}."
            )

        if self.slice_obj.step not in (None, 1):
            raise ValueError(
                "Sliced patch commit currently only supports contiguous slices with step None or 1. "
                f"Got step={self.slice_obj.step}."
            )

        axis_size = orig_raw.shape[ax_idx]

        # Normalize Python slice semantics, including negative bounds.
        start, stop, step = self.slice_obj.indices(axis_size)

        if step != 1:
            raise ValueError(
                "Sliced patch commit currently only supports contiguous forward slices with step 1. "
                f"Got normalized step={step}."
            )

        if start > stop:
            raise ValueError(
                f"Invalid patch slice bounds after normalization: start={start}, stop={stop}, "
                f"axis_size={axis_size}."
            )

        expected_patch_len = stop - start
        actual_patch_len = chunk_raw.shape[ax_idx]

        if actual_patch_len != expected_patch_len:
            raise ValueError(
                "Cannot commit sliced patch because the chunk length no longer matches the target slice. "
                f"Expected length {expected_patch_len} on axis '{self.target_ax.name}', "
                f"got {actual_patch_len}."
            )

        left_slice = [slice(None)] * orig_raw.ndim
        left_slice[ax_idx] = slice(None, start)

        right_slice = [slice(None)] * orig_raw.ndim
        right_slice[ax_idx] = slice(stop, None)

        left_raw = orig_raw[tuple(left_slice)]
        right_raw = orig_raw[tuple(right_slice)]

        stitched_raw = jnp.concatenate([left_raw, chunk_raw, right_raw], axis=ax_idx)

        return Tensor(stitched_raw, *self.original_tensor.topology)


class TargetedSlicedMonad:
    """
    Targeted view into a SlicedMonad's chunk.

    This is what makes things like:

        x.s[10:20].d.bias()[:]

    preserve the patch context instead of accidentally decaying to a plain Tensor.
    """

    def __init__(self, monad: SlicedMonad, target_axes: Tuple[Axis, ...]):
        self.monad = monad
        self.target_axes = target_axes

    @property
    def tensor(self):
        return self.monad.chunk_tensor

    def _targeted(self):
        return TargetedTensor(self.monad.chunk_tensor, self.target_axes)

    def _wrap(
        self,
        result,
        *,
        patch_safe: Optional[bool] = None,
        unsafe_reason: Optional[str] = None,
    ):
        return self.monad._wrap(
            result,
            patch_safe=patch_safe,
            unsafe_reason=unsafe_reason,
        )

    def __getattr__(self, name: str):
        # Chain another axis while preserving patch context.
        for axis in self.monad.chunk_tensor.topology:
            if axis.name == name:
                if axis not in self.target_axes:
                    return TargetedSlicedMonad(self.monad, self.target_axes + (axis,))
                return self

        # Dynamic NN lookup / TargetedTensor methods.
        attr = getattr(self._targeted(), name)

        if callable(attr):
            def wrapped(*args, **kwargs):
                return self._wrap(attr(*args, **kwargs))
            return wrapped

        return attr

    def pw(self, func, tie: Optional[str] = None, **kwargs):
        return self._wrap(self._targeted().pw(func, tie=tie, **kwargs))

    def proj(self, *target_axes, bias: bool = False, tie: Optional[str] = None, init=None):
        explicit_projection = len(target_axes) > 0

        result = self._targeted().proj(
            *target_axes,
            bias=bias,
            tie=tie,
            init=init,
        )

        if explicit_projection:
            return self._wrap(
                result,
                patch_safe=False,
                unsafe_reason=(
                    "Explicit proj(...) inside a sliced patch is not commit-safe. "
                    "Use proj() with no explicit axes for same-axis patch projection, "
                    "or stitch manually."
                ),
            )

        return self._wrap(result)

    def bias(self, init=None, tie: Optional[str] = None):
        return self._wrap(self._targeted().bias(init=init, tie=tie))

    def gate(self, init=None, tie: Optional[str] = None):
        return self._wrap(self._targeted().gate(init=init, tie=tie))

    def mask(self, func, fill: float):
        return self._wrap(self._targeted().mask(func, fill=fill))

    def rename(self, new_axis: Axis):
        result = self._targeted().rename(new_axis)
        return self._wrap(
            result,
            patch_safe=False,
            unsafe_reason=(
                "rename(...) inside a sliced patch is not commit-safe. "
                "Rename or stitch manually outside the patch monad."
            ),
        )

    def pad(self, *pad_widths, fill: float = 0.0):
        result = self._targeted().pad(*pad_widths, fill=fill)
        return self._wrap(
            result,
            patch_safe=False,
            unsafe_reason=(
                "pad(...) inside a sliced patch is not commit-safe because it changes topology/size. "
                "Pad or stitch manually outside the patch monad."
            ),
        )

    def unfold(self, window_axis: Axis, step: int = 1):
        result = self._targeted().unfold(window_axis, step=step)
        return self._wrap(
            result,
            patch_safe=False,
            unsafe_reason=(
                "unfold(...) inside a sliced patch is not commit-safe because it changes topology. "
                "Unfold or stitch manually outside the patch monad."
            ),
        )

    def sum(self):
        # Reductions are value-style. They do not preserve patch context.
        return self._targeted().sum()

    def mean(self):
        return self._targeted().mean()

    def max(self):
        return self._targeted().max()

    @property
    def size(self) -> int:
        return self._targeted().size

    def __int__(self) -> int:
        return int(self._targeted())

    def __index__(self) -> int:
        return self._targeted().__index__()

    def __getitem__(self, key):
        result = self._targeted().__getitem__(key)

        # Nested slicing inside a patch is allowed as value-style behavior,
        # but it usually will not be commit-compatible unless topology remains identical.
        return self._wrap(result)

    def __matmul__(self, other):
        if hasattr(other, "chunk_tensor"):
            other = other.chunk_tensor
        return self._targeted().__matmul__(other)


class TargetedTensor(NNTargetedTensorStubs):
    """The transient object returned when targeting axes (e.g., x.d)"""

    def __init__(self, tensor: 'Tensor', target_axes: Tuple[Axis, ...]):
        self.tensor = tensor
        self.target_axes = target_axes

    def __getattr__(self, name: str) -> Any:
        """Chains another target axis OR dynamically invokes an nn.py function!"""
        # 1. Check if they are chaining an axis (e.g., x.b.s)
        for axis in self.tensor.topology:
            if axis.name == name:
                if axis not in self.target_axes:
                    return TargetedTensor(self.tensor, self.target_axes + (axis,))
                return self

        # Dynamic NN Library Lookup
        # We import locally to prevent circular imports between core.py and nn.py
        from . import nn
        # Prevent loss functions from being dynamically chained
        if hasattr(nn, name) and not name.endswith('_loss') and not name.endswith('_logits'):
            func = getattr(nn, name)
            if callable(func):
                # We return a bound lambda that automatically routes the function into .pw()!
                def bound_nn_method(*args, **kwargs):
                    return self.pw(func, *args, **kwargs)

                return bound_nn_method

        raise AttributeError(
            f"Axis or NN function '{name}' not found. Available axes: {[a.name for a in self.tensor.topology]}")

    def pw(self, func, tie: Optional[str] = None, **kwargs) -> 'Tensor':
        import jax
        import inspect

        # 1. Axiom-Native NN Modules
        # If the function is from axiom.nn, pass the whole TargetedTensor to it
        if getattr(func, '_is_axiom_nn', False):
            return func(self, tie=tie, **kwargs)

        # 2. Standard JAX Functions (jnp.exp, jax.nn.sigmoid, jax.nn.softmax)
        # If the JAX function accepts an 'axis' argument, automatically inject the targeted axes!
        try:
            sig = inspect.signature(func)
            if 'axis' in sig.parameters:
                axis_indices = tuple(self.tensor.topology.index(a) for a in self.target_axes)
                kwargs['axis'] = axis_indices if len(axis_indices) > 1 else axis_indices[0]
        except ValueError:
            pass  # Built-in C functions might not have accessible signatures

        raw_result = func(self.tensor.unwrap(), **kwargs)
        return Tensor(raw_result, *self.tensor.topology)

    def proj(self, *target_axes: 'Axis', bias: bool = False, tie: Optional[str] = None, init=None) -> 'Tensor':
        if not target_axes:
            target_axes = self.target_axes

        import jax.numpy as jnp, numpy as np
        from .state import state
        from . import init as ax_init

        in_dim = np.prod([a.size for a in self.target_axes])
        out_dim = np.prod([a.size for a in target_axes])
        if in_dim is None or out_dim is None:
            raise ValueError("Projections require axes to have statically defined sizes.")

        # Default to Xavier if no init is provided
        initializer = init if init is not None else ax_init.xavier

        # Get/Init the Weight Matrix via the State Manager
        W_raw = state.get_param(
            layer_type="proj_w",
            shape=(in_dim, out_dim),
            init_fn=initializer,
            tie=tie,
            fan_in=in_dim,
            fan_out=out_dim
        )
        W_param = Tensor(W_raw, Axis("_in", in_dim), Axis("_out", out_dim))

        # --- Topologically-Aware Flattening (Your existing flawless logic) ---
        kept_axes = tuple(a for a in self.tensor.topology if a not in self.target_axes)
        transpose_order = [self.tensor.topology.index(a) for a in kept_axes + self.target_axes]
        transposed_raw = jnp.transpose(self.tensor.unwrap(), transpose_order)

        kept_shape = tuple(a.size if a.size is not None else -1 for a in kept_axes)
        flattened_raw = jnp.reshape(transposed_raw, kept_shape + (in_dim,))

        result_raw = jnp.dot(flattened_raw, W_param.unwrap())

        new_topology = kept_axes + target_axes
        new_shape = tuple(a.size if a.size is not None else -1 for a in new_topology)
        result_raw = jnp.reshape(result_raw, new_shape)
        result_tensor = Tensor(result_raw, *new_topology)

        # --- BIAS LOGIC ---
        if bias:
            # We explicitly use zeros here, but we target the new output axes
            return result_tensor.target(*target_axes).bias(tie=f"{tie}_bias" if tie else None)

        return result_tensor

    def bias(self, init=None, tie: Optional[str] = None) -> 'Tensor':
        """Adds a learnable bias parameter matching the targeted axes."""
        from .state import state
        from . import init as ax_init

        initializer = init if init is not None else ax_init.zeros
        shape = tuple(a.size for a in self.target_axes)

        b_raw = state.get_param("bias", shape, initializer, tie=tie)
        b_tensor = Tensor(b_raw, *self.target_axes)

        # Native Axiom addition will automatically broadcast over the untargeted axes!
        return self.tensor + b_tensor

    def gate(self, init=None, tie: Optional[str] = None) -> 'Tensor':
        """Multiplies the tensor by a learnable scaling parameter matching the targeted axes."""
        from .state import state
        from . import init as ax_init

        initializer = init if init is not None else ax_init.ones
        shape = tuple(a.size for a in self.target_axes)

        g_raw = state.get_param("gate", shape, initializer, tie=tie)
        g_tensor = Tensor(g_raw, *self.target_axes)

        # Native Axiom multiplication automatically broadcasts
        return self.tensor * g_tensor

    def mask(self, func, fill: float) -> 'Tensor':
        """Native coordinate-based masking."""
        import jax.numpy as jnp

        # 1. Generate 1D coordinates for targeted axes
        grids = []
        for a in self.target_axes:
            if a.size is None:
                raise ValueError(f"Masking requires sized axes. '{a.name}' is missing a size.")
            grids.append(jnp.arange(a.size))

        # 2. Create the interaction grid
        mesh = jnp.meshgrid(*grids, indexing='ij')
        bool_mask = func(*mesh)

        # 3. Align mask to the parent tensor's topology and apply
        mask_tensor = Tensor(bool_mask, *self.target_axes)
        aligned_mask = mask_tensor._align_to(self.tensor.topology)
        result_raw = jnp.where(aligned_mask, fill, self.tensor.unwrap())

        return Tensor(result_raw, *self.tensor.topology)

    def unfold(self, window_axis: Axis, step: int = 1) -> 'Tensor':
        """Creates a sliding window over a spatial sequence for convolutions."""
        import jax.numpy as jnp

        if len(self.target_axes) != 1:
            raise ValueError("Unfold requires exactly one target axis (e.g., x.seq.unfold(...)).")

        spatial_ax = self.target_axes[0]
        if spatial_ax.size is None or window_axis.size is None:
            raise ValueError("Unfold requires both the target axis and the window axis to have defined sizes.")

        kernel_size = window_axis.size
        spatial_size = spatial_ax.size
        out_size = (spatial_size - kernel_size) // step + 1

        if out_size <= 0:
            raise ValueError("Kernel size cannot be larger than the spatial dimension.")

        # 1. Generate sliding window indices: shape (out_size, kernel_size)
        starts = jnp.arange(0, out_size * step, step)[:, None]
        offsets = jnp.arange(kernel_size)[None, :]
        indices = starts + offsets

        raw_x = self.tensor.unwrap()
        ax_idx = self.tensor.topology.index(spatial_ax)

        # 2. Gather the windows natively
        # jnp.take replaces the targeted axis with the new 2D index shape!
        unfolded_raw = jnp.take(raw_x, indices, axis=ax_idx)

        # 3. Compute the new topology
        # The old spatial axis is replaced by the (New Spatial Axis, Kernel Axis)
        new_spatial_ax = Axis(spatial_ax.name, out_size)

        new_topology = list(self.tensor.topology)
        new_topology.pop(ax_idx)
        new_topology.insert(ax_idx, window_axis)
        new_topology.insert(ax_idx, new_spatial_ax)

        return Tensor(unfolded_raw, *new_topology)

    def scan(self, func, init: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        """Recurrent scan over the targeted axis."""
        import jax
        import jax.numpy as jnp

        if len(self.target_axes) != 1:
            raise ValueError("Scan requires exactly one target axis (e.g., x.s.scan(...)).")
        scan_ax = self.target_axes[0]

        raw_x = self.tensor.unwrap()
        ax_idx = self.tensor.topology.index(scan_ax)

        # JAX requires the scanned axis to be at index 0. We transpose it.
        perm = [ax_idx] + [i for i in range(len(self.tensor.topology)) if i != ax_idx]
        inv_perm = [perm.index(i) for i in range(len(self.tensor.topology))]
        xs_transposed = jnp.transpose(raw_x, perm)

        # The topology of the elements inside the loop (lacks the scan axis)
        xt_topology = tuple(a for a in self.tensor.topology if a != scan_ax)

        def wrapped_func(raw_carry, raw_xt):
            # Wrap raw arrays back into Axiom Tensors for the user's lambda
            carry_tensor = Tensor(raw_carry, *init.topology)
            xt_tensor = Tensor(raw_xt, *xt_topology)

            new_carry, out_y = func(carry_tensor, xt_tensor)
            return new_carry.unwrap(), out_y.unwrap()

        final_carry_raw, y_seq_raw = jax.lax.scan(wrapped_func, init.unwrap(), xs_transposed)

        # Transpose the sequence output back to the original topological order
        y_seq_raw = jnp.transpose(y_seq_raw, inv_perm)

        final_carry = Tensor(final_carry_raw, *init.topology)
        y_seq = Tensor(y_seq_raw, *self.tensor.topology)

        return final_carry, y_seq

    def pad(self, *pad_widths: Tuple[int, int], fill: float = 0.0) -> 'Tensor':
        """Native topological padding. e.g., x.seq.pad((1, 2)) or x.h.w.pad((1,1), (2,2))"""
        import jax.numpy as jnp

        if len(pad_widths) != len(self.target_axes):
            # QoL: If they provide one tuple but target multiple axes, broadcast it automatically!
            if len(pad_widths) == 1:
                pad_widths = pad_widths * len(self.target_axes)
            else:
                raise ValueError(f"Provided {len(pad_widths)} pad widths for {len(self.target_axes)} targeted axes.")

        # Build the JAX pad_width sequence [(0,0), ..., (0,0)]
        pad_width_full = [(0, 0)] * len(self.tensor.topology)
        new_topology = list(self.tensor.topology)

        # Inject the paddings and update topology sizes
        for target, (before, after) in zip(self.target_axes, pad_widths):
            ax_idx = self.tensor.topology.index(target)
            pad_width_full[ax_idx] = (before, after)

            new_size = (target.size + before + after) if target.size is not None else None
            new_topology[ax_idx] = Axis(target.name, new_size)

        padded_raw = jnp.pad(self.tensor.unwrap(), pad_width_full, constant_values=fill)
        return Tensor(padded_raw, *new_topology)

    def rename(self, new_axis: Axis) -> 'Tensor':
        """
        Renames the targeted axis without altering data.
        e.g., x.d2.rename(ax.d)
        """
        if len(self.target_axes) != 1:
            raise ValueError("Rename requires exactly one target axis (e.g., x.d2.rename(ax.d)).")

        target = self.target_axes[0]
        ax_idx = self.tensor.topology.index(target)

        # STRICTNESS GUARD: Prevent silent reshaping!
        if new_axis.size is not None and new_axis.size != target.size:
            raise ValueError(
                f"Topological Violation: .rename() cannot change axis sizes. "
                f"Tried to rename '{target.name}' (size {target.size}) to "
                f"'{new_axis.name}' (size {new_axis.size}). Use .proj() to change sizes."
            )

        # Force the new axis to strictly inherit the existing physical size
        final_axis = Axis(new_axis.name, target.size)

        new_topology = list(self.tensor.topology)
        new_topology[ax_idx] = final_axis

        return Tensor(self.tensor.unwrap(), *new_topology)

    def _reduce(self, jnp_func) -> 'Tensor':
        """Helper for reduction operations (sum, mean, max)"""
        # Find the integer indices of the targeted axes
        axis_indices = tuple(self.tensor.topology.index(a) for a in self.target_axes)

        # Execute the raw JAX reduction
        raw_result = jnp_func(self.tensor.unwrap(), axis=axis_indices)

        # Compute the new topology (targeted axes are collapsed/removed)
        new_topology = tuple(a for a in self.tensor.topology if a not in self.target_axes)
        return Tensor(raw_result, *new_topology)

    def sum(self) -> 'Tensor':
        import jax.numpy as jnp
        return self._reduce(jnp.sum)

    def mean(self) -> 'Tensor':
        import jax.numpy as jnp
        return self._reduce(jnp.mean)

    def max(self) -> 'Tensor':
        import jax.numpy as jnp
        return self._reduce(jnp.max)

    @property
    def size(self) -> int:
        """Returns the size of the targeted axis (or product of axes if multiple)."""
        import numpy as np
        sizes = [a.size for a in self.target_axes]
        if None in sizes:
            raise ValueError("Targeted axes must have defined sizes.")
        return int(np.prod(sizes))

    def __int__(self) -> int:
        return self.size

    def __index__(self) -> int:
        """This is the method JAX looks for to cast objects to shape integers!"""
        return self.size

    # --- TRANSPARENT MATH PROXIES ---
    def __mul__(self, other):
        if isinstance(other, int): return self.size * other
        return self.tensor * other

    def __rmul__(self, other):
        return self.tensor * other

    def __floordiv__(self, other):
        if isinstance(other, int): return self.size // other
        return self.tensor // other

    def __add__(self, other):
        if isinstance(other, int): return self.size + other
        return self.tensor + other

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
        """Explicit Multi-Axis Contraction (e.g., scores.sk1.sk2 @ v)"""
        if not hasattr(other, 'topology'):
            raise ValueError("Matrix multiplication requires another Tensor.")

        # 1. Verify the targeted axes actually exist in the right tensor
        for target_ax in self.target_axes:
            if target_ax not in other.topology:
                raise ValueError(
                    f"Cannot contract over '{target_ax.name}': axis missing in right tensor. "
                    f"Right topology: {[a.name for a in other.topology]}"
                )

        # 2. Native broadcasting multiplication (Aligns shared batches, expands the rest!)
        result = self.tensor * other

        # 3. Sum over ONLY the targeted axes
        for target_ax in self.target_axes:
            result = getattr(result, target_ax.name).sum()

        return result

    def __getitem__(self, key: Any) -> Any:
        if key == slice(None):
            return self.tensor

        # If a lazy slice is used as an index, consume it as its sliced Tensor.
        # This makes E.v[data.s[:-1]].gather() work.
        if hasattr(key, "chunk_tensor"):
            key = key.chunk_tensor

        if hasattr(key, 'topology'):
            return RoutedContext(self.tensor, self.target_axes, key)

        if len(self.target_axes) != 1:
            raise ValueError("Slicing requires exactly one target axis.")

        target = self.target_axes[0]
        ax_idx = self.tensor.topology.index(target)

        full_slice = [slice(None)] * len(self.tensor.topology)
        full_slice[ax_idx] = key
        sliced_raw = self.tensor.unwrap()[tuple(full_slice)]

        if isinstance(key, int):
            # Integer indexing physically removes the dimension from the array.
            new_topology = list(self.tensor.topology)
            new_topology.pop(ax_idx)
            return Tensor(sliced_raw, *new_topology)

        new_size = sliced_raw.shape[ax_idx]
        new_axis = Axis(target.name, new_size)

        new_topology = list(self.tensor.topology)
        new_topology[ax_idx] = new_axis
        chunk_tensor = Tensor(sliced_raw, *new_topology)

        # Non-integer slicing creates a lazy slice / optional patch transaction.
        return SlicedMonad(
            self.tensor,
            target,
            key,
            chunk_tensor,
            expected_topology=chunk_tensor.topology,
            patch_safe=True,
        )

    def __dir__(self):
        """Exposes dynamic NN functions to dynamic autocomplete (Jupyter/REPL)."""
        from . import nn
        base_dir = super().__dir__()

        # Add all available axes
        axes = [a.name for a in (self.tensor.topology if hasattr(self, 'tensor') else self._axes)]

        # Add all public callable functions from your nn library!
        nn_funcs = [k for k, v in vars(nn).items() if callable(v) and not k.startswith('_')]

        return sorted(set(base_dir + axes + nn_funcs))


class TargetedBundle(NNTargetedBundleStubs):
    """Handles operations targeted across multiple tensors simultaneously."""
    def __init__(self, bundle: 'Bundle', target_axes: Tuple[Axis, ...]):
        self.bundle = bundle
        self.target_axes = target_axes

    def __getattr__(self, name: str) -> Any:
        """Chains another target axis OR dynamically invokes an nn.py function across the bundle!"""
        # Check if they are chaining an axis
        for tensor in self.bundle.tensors:
            for axis in tensor.topology:
                if axis.name == name:
                    if axis not in self.target_axes:
                        return TargetedBundle(self.bundle, self.target_axes + (axis,))
                    return self

        # Dynamic NN Library Lookup for Bundles
        from . import nn
        # Prevent loss functions from being dynamically chained
        if hasattr(nn, name) and not name.endswith('_loss') and not name.endswith('_logits'):
            func = getattr(nn, name)
            if callable(func):
                def bound_nn_method(*args, **kwargs):
                    return self.pw(func, *args, **kwargs)

                return bound_nn_method

        raise AttributeError(f"Axis or NN function '{name}' not found in bundled tensors.")

    def proj(self, *target_axes: 'Axis', bias: bool = False, tie: Optional[str] = None, init=None) -> 'Bundle':
        """Parallel Bundle Projection. Returns a chainable Bundle."""
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.proj(*target_axes, bias=bias, tie=tie, init=init))
        return Bundle(*results)  # Return a Bundle for infinite chaining!

    def bias(self, init=None, tie: Optional[str] = None) -> 'Bundle':
        """Parallel Bias Application. Returns a chainable Bundle."""
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.bias(init=init, tie=tie))
        return Bundle(*results)

    def gate(self, init=None, tie: Optional[str] = None) -> 'Bundle':
        """Parallel Gate Application. Returns a chainable Bundle."""
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.gate(init=init, tie=tie))
        return Bundle(*results)

    def pw(self, func, tie: Optional[str] = None, **kwargs) -> Tuple['Tensor', ...]:
        """Parallel Pointwise Mapping."""
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.pw(func, tie=tie, **kwargs))
        return tuple(results)

    def unfold(self, window_axis: 'Axis', step: int = 1) -> 'Bundle':
        """Parallel topological unfolding across the bundle. Returns a chainable Bundle."""
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.unfold(window_axis, step=step))
        return Bundle(*results)

    def assoc_scan(self, func) -> Tuple['Tensor', ...]:
        """Parallel associative scan over the bundled tensors."""
        import jax
        import jax.numpy as jnp

        if len(self.target_axes) != 1:
            raise ValueError("Associative scan requires exactly one target axis.")
        scan_ax = self.target_axes[0]

        raw_elems = []
        inv_perms = []
        inner_topologies = []

        for tensor in self.bundle.tensors:
            if scan_ax not in tensor.topology:
                raise ValueError(f"Axis '{scan_ax.name}' missing from a bundled tensor.")

            idx = tensor.topology.index(scan_ax)
            perm = [idx] + [i for i in range(len(tensor.topology)) if i != idx]
            inv_perms.append([perm.index(i) for i in range(len(tensor.topology))])
            raw_elems.append(jnp.transpose(tensor.unwrap(), perm))

            # Base topology (without the sequence axis)
            inner_topologies.append(tuple(a for a in tensor.topology if a != scan_ax))

        def wrapped_func(raw_left, raw_right):
            left_tensors = []
            right_tensors = []
            chunk_ax = None

            for r_l, r_r, top in zip(raw_left, raw_right, inner_topologies):
                current_top = top

                # JAX Magic Detection:
                # If JAX passes a sequence chunk, we dynamically attach a '_chunk' axis.
                # If it's inside an internal vmap (BatchTracer), the shape aligns natively.
                if hasattr(r_l, 'shape') and len(r_l.shape) == len(top) + 1:
                    if chunk_ax is None:
                        chunk_ax = Axis("_chunk", r_l.shape[0])
                    current_top = (chunk_ax,) + top

                left_tensors.append(Tensor(r_l, *current_top))
                right_tensors.append(Tensor(r_r, *current_top))

            out_tensors = func(tuple(left_tensors), tuple(right_tensors))
            return tuple(out.unwrap() for out in out_tensors)

        # Execute the parallel associative scan
        out_raw_elems = jax.lax.associative_scan(wrapped_func, tuple(raw_elems))

        # Transpose back and wrap to original topologies
        out_tensors = []
        for raw_out, inv_perm, orig_tensor in zip(out_raw_elems, inv_perms, self.bundle.tensors):
            restored = jnp.transpose(raw_out, inv_perm)
            out_tensors.append(Tensor(restored, *orig_tensor.topology))

        return tuple(out_tensors)

    def rename(self, *new_axes: Axis) -> Tuple['Tensor', ...]:
        """
        Parallel topological renaming across the bundle.
        e.g., (x & y).s.rename(ax.new_s) -> Both get new_s
        e.g., (x & y).s.rename(ax.sx, ax.sy) -> x gets sx, y gets sy
        """
        if len(new_axes) == 1:
            # Backward Compatibility: Broadcast 1 axis to all tensors
            new_axes = new_axes * len(self.bundle.tensors)
        elif len(new_axes) != len(self.bundle.tensors):
            raise ValueError(
                f"Bundle contains {len(self.bundle.tensors)} tensors, "
                f"but {len(new_axes)} axes were provided to rename()."
            )

        results = []
        for tensor, new_axis in zip(self.bundle.tensors, new_axes):
            t_tensor = TargetedTensor(tensor, self.target_axes)
            # Reuses the strict size-checking logic inside TargetedTensor!
            results.append(t_tensor.rename(new_axis))

        return tuple(results)

    def join(self) -> 'Tensor':
        """
        Concatenates the bundled tensors along the targeted axis.
        Returns a single Tensor. e.g., (x & y).d.join()
        """
        import jax.numpy as jnp

        if len(self.target_axes) != 1:
            raise ValueError("Join requires exactly one target axis (e.g., (x & y).d.join()).")

        target_ax = self.target_axes[0]

        # Use the first tensor as the baseline for the topological layout
        base_tensor = self.bundle.tensors[0]
        ax_idx = base_tensor.topology.index(target_ax)

        raw_arrays = []
        total_size = 0

        for tensor in self.bundle.tensors:
            # Extract the actual physical axis from this specific tensor
            # (Because x might have d(32) and h might have d(64))
            current_ax = tensor.topology[tensor.topology.index(target_ax)]

            if current_ax.size is None:
                raise ValueError(f"Cannot join on axis '{target_ax.name}' without a defined size.")

            total_size += current_ax.size
            raw_arrays.append(tensor.unwrap())

        # 1. Native JAX Concatenation
        # (JAX will automatically throw a native shape error here if the
        # NON-targeted axes don't match, which is exactly the behavior we want!)
        joined_raw = jnp.concatenate(raw_arrays, axis=ax_idx)

        # 2. Reconstruct the topology with the newly summed dimension!
        new_topology = list(base_tensor.topology)
        new_topology[ax_idx] = Axis(target_ax.name, total_size)

        return Tensor(joined_raw, *new_topology)

    def pad(self, *pad_widths: Tuple[int, int], fill: float = 0.0) -> Tuple['Tensor', ...]:
        """Parallel topological padding across the bundle."""
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.pad(*pad_widths, fill=fill))
        return tuple(results)

    def mask(self, func, fill: float = 0.0) -> Tuple['Tensor', ...]:
        """Parallel topological masking across the bundle."""
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.mask(func, fill=fill))
        return tuple(results)

    def __dir__(self):
        """Exposes dynamic NN functions to dynamic autocomplete (Jupyter/REPL)."""
        from . import nn
        base_dir = super().__dir__()

        # Collect ALL unique axes across every tensor in the bundle
        axes = set()
        for tensor in self.bundle.tensors:
            for axis in tensor.topology:
                axes.add(axis.name)

        # Add all public callable functions from your nn library
        nn_funcs = [k for k, v in vars(nn).items() if callable(v) and not k.startswith('_')]

        return sorted(set(base_dir + list(axes) + nn_funcs))


class Bundle:
    """Wraps multiple Tensors to perform parallel, fused operations."""
    def __init__(self, *tensors: Tensor):
        self.tensors = tensors

    def __and__(self, other: 'Tensor') -> 'Bundle':
        return Bundle(*self.tensors, other)

    def __iter__(self):
        """Allows direct unpacking: x, h = Bundle(x, h)"""
        return iter(self.tensors)

    def __getattr__(self, name: str) -> 'TargetedBundle':
        # Initialize targeting
        for tensor in self.tensors:
            for axis in tensor.topology:
                if axis.name == name:
                    return TargetedBundle(self, (axis,))
        raise AttributeError(f"Axis '{name}' not found in bundled tensors.")

    def apply_n(self, func: callable, times: int) -> 'Bundle':
        """
        Recursively applies a function `times` times across bundled states using jax.lax.scan.
        Perfect for RNNs, LSTMs, and multi-state State-Space Models.
        """
        import jax.lax as lax

        def _scan_body(carry_raw_tuple, _):
            # 1. Reconstruct the Axiom Tensors from the raw JAX arrays
            carry_tensors = [
                Tensor(raw, *orig_t.topology)
                for raw, orig_t in zip(carry_raw_tuple, self.tensors)
            ]

            # 2. Rebuild the Bundle and apply the user's function
            carry_bundle = Bundle(*carry_tensors)
            out_bundle = func(carry_bundle)

            # Type and structural checks
            if not isinstance(out_bundle, Bundle):
                raise TypeError(
                    f"apply_n on a Bundle requires the function to return a Bundle. Got {type(out_bundle)}.")

            if len(out_bundle.tensors) != len(self.tensors):
                raise ValueError(
                    f"Function returned {len(out_bundle.tensors)} tensors, "
                    f"but the input bundle had {len(self.tensors)}."
                )

            # 3. Topology check for XLA stability (Input and Output topologies MUST match)
            out_raw_tuple = []
            for in_t, out_t in zip(self.tensors, out_bundle.tensors):
                if in_t.topology != out_t.topology:
                    raise ValueError(
                        f"apply_n requires matched topologies for XLA. "
                        f"Input tensor had {[a.name for a in in_t.topology]}, "
                        f"but output tensor had {[a.name for a in out_t.topology]}."
                    )
                out_raw_tuple.append(out_t.unwrap())

            # 4. Return the tuple of raw arrays for the hardware loop
            return tuple(out_raw_tuple), None

        # Extract the raw arrays to pass into lax.scan
        init_raw_tuple = tuple(t.unwrap() for t in self.tensors)

        # Execute the hardware-level loop (length=times acts as our depth)
        final_raw_tuple, _ = lax.scan(_scan_body, init_raw_tuple, None, length=times)

        # Rewrap the final output into a new Bundle
        final_tensors = [
            Tensor(raw, *orig_t.topology)
            for raw, orig_t in zip(final_raw_tuple, self.tensors)
        ]
        return Bundle(*final_tensors)


class Tensor(NNTensorStubs):
    """The core Axiom Tensor wrapper enforcing named-axis topologies."""

    def __init__(self, raw_tensor: Any, *axes: Axis):
        self._tensor = raw_tensor
        self._axes = axes
        self._is_param = False
        self._param_name = None
        self._validate_topology()

    def _validate_topology(self):
        if hasattr(self._tensor, 'shape'):
            if len(self._tensor.shape) != len(self._axes):
                raise ValueError(
                    f"Topology mismatch: Array has {len(self._tensor.shape)} dims, given {len(self._axes)} axes.")
            for i, (dim_size, axis) in enumerate(zip(self._tensor.shape, self._axes)):
                if axis.size is not None and axis.size != dim_size:
                    raise ValueError(f"Size mismatch on '{axis.name}': expected {axis.size}, got {dim_size}.")

    def unwrap(self) -> Any:
        return self._tensor

    @property
    def topology(self) -> Tuple[Axis, ...]:
        return self._axes

    def param(self, name: Optional[str] = None, tie: Optional[Tie] = None) -> 'Tensor':
        """Registers the tensor as a trainable parameter for the AOT compiler."""
        self._is_param = True

        p_name = tie.name if tie else name
        if p_name is None:
            p_name = f"param_{compiler_state.param_counter}"

        compiler_state.param_counter += 1
        self._param_name = p_name

        # If tracing, allocate the param. If applying, retrieve it!
        if compiler_state.is_initializing:
            if p_name not in compiler_state.params:
                compiler_state.params[p_name] = self.unwrap()
            return Tensor(compiler_state.params[p_name], *self.topology)
        elif compiler_state.params and p_name in compiler_state.params:
            return Tensor(compiler_state.params[p_name], *self.topology)

        return self  # Eager execution fallback

    def pw(self, func, **kwargs) -> 'Tensor':
        """
        Global Pointwise Mapping.
        Applies an element-wise function across the entire tensor.
        """
        raw_result = func(self.unwrap(), **kwargs)
        return Tensor(raw_result, *self._axes)

    def apply_n(self, func: callable, times: int) -> 'Tensor':
        """
        Recursively applies a function `times` times using jax.lax.scan.
        Guarantees O(1) compilation time and strictly tied hardware weights.
        """
        import jax.lax as lax

        def _scan_body(carry_raw, _):
            # 1. Reconstruct the Axiom Tensor for the user's function
            carry_tensor = Tensor(carry_raw, *self.topology)

            # 2. Apply the block function
            out_tensor = func(carry_tensor)

            # 3. XLA scan requires the carry to remain identical in shape/dtype
            if out_tensor.topology != carry_tensor.topology:
                raise ValueError(
                    f"apply_n requires the function to return the same topology. "
                    f"Input was {[a.name for a in carry_tensor.topology]}, "
                    f"but function returned {[a.name for a in out_tensor.topology]}."
                )

            # 4. Unwrap back to a raw JAX array for the hardware loop
            return out_tensor.unwrap(), None

        # Execute the hardware-level loop (length=times acts as our depth)
        final_raw, _ = lax.scan(_scan_body, self.unwrap(), None, length=times)

        # Rewrap the final output
        return Tensor(final_raw, *self.topology)

    # ==========================================
    # UNION-BASED TOPOLOGY ENGINE
    # ==========================================
    def _get_union_topology(self, other: 'Tensor') -> Tuple[Axis, ...]:
        union = list(self._axes)
        for ax_other in other.topology:
            if ax_other not in union:
                union.append(ax_other)

        # Collision Check
        for u in union:
            sz_a = next((a.size for a in self._axes if a == u), None)
            sz_b = next((b.size for b in other.topology if b == u), None)
            if sz_a is not None and sz_b is not None and sz_a != sz_b:
                raise ValueError(f"TopologyCollision: Axis '{u.name}' has conflicting sizes {sz_a} and {sz_b}.")
        return tuple(union)

    def _align_to(self, target_axes: Tuple[Axis, ...]) -> Any:
        import jax.numpy as jnp
        raw = self.unwrap()

        # 1. Transpose existing axes to match target order
        current_indices = [self._axes.index(ta) for ta in target_axes if ta in self._axes]
        transposed = jnp.transpose(raw, current_indices)

        # 2. Inject 1s for missing axes to trigger JAX broadcasting
        final_shape = [ta.size if ta in self._axes else 1 for ta in target_axes]
        return jnp.reshape(transposed, final_shape)


    def _broadcast_op(self, other, op_func) -> 'Tensor':
        # Auto-unwrap SlicedMonads and TargetedTensors seamlessly!
        if hasattr(other, 'chunk_tensor'):
            other = other.chunk_tensor
        elif hasattr(other, 'tensor') and hasattr(other, 'target_axes'):
            other = other.tensor

        if not isinstance(other, Tensor):
            return Tensor(op_func(self.unwrap(), other), *self._axes)

        union_axes = self._get_union_topology(other)
        aligned_self = self._align_to(union_axes)
        aligned_other = other._align_to(union_axes)
        return Tensor(op_func(aligned_self, aligned_other), *union_axes)


    # Dunder Math Methods
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

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Implicit Matrix Multiplication (Contracts the right-most shared axis)"""
        if not hasattr(other, 'topology'):
            raise ValueError("Matrix multiplication requires another Tensor.")

        # 1. Find all shared axes
        shared_axes = [a for a in self.topology if a in other.topology]
        if not shared_axes:
            raise ValueError(
                f"No shared axes to contract between "
                f"{[a.name for a in self.topology]} and {[a.name for a in other.topology]}."
            )

        # 2. The Implicit Rule: Pick the right-most shared axis from THIS tensor's topology
        contract_ax = shared_axes[-1]

        # 3. Route it explicitly using our new TargetedTensor logic!
        return TargetedTensor(self, (contract_ax,)) @ other

    def __getattr__(self, name: str) -> Any:
        """Targets an axis OR dynamically invokes a purely pointwise nn function!"""
        # 1. Axis Targeting (e.g., x.d)
        for axis in self._axes:
            if axis.name == name:
                return TargetedTensor(self, (axis,))

        # 2. Pure Pointwise NN Lookup
        from . import nn
        import inspect

        # Prevent loss functions from being dynamically chained
        if hasattr(nn, name) and not name.endswith('_loss') and not name.endswith('_logits'):
            func = getattr(nn, name)
            if callable(func):

                # STRICTNESS GUARD: Does this function require an axis?
                # We check the signature, or catch known custom stateful modules.
                requires_axis = False
                try:
                    sig = inspect.signature(func)
                    if 'axis' in sig.parameters:
                        requires_axis = True
                except ValueError:
                    pass

                if requires_axis or getattr(func, '_is_axiom_nn', False):
                    raise ValueError(
                        f"Mathematical Ambiguity: '{name}' requires a target axis. "
                        f"You must target an axis first! (e.g., x.d.{name}())"
                    )

                # It's a pure pointwise function! Execute directly on the unwrapped array.
                def bound_pointwise(*args, **kwargs):
                    raw_res = func(self.unwrap(), *args, **kwargs)
                    return Tensor(raw_res, *self.topology)

                return bound_pointwise

        raise AttributeError(f"Tensor has no axis or pure function '{name}'. Topology: {[a.name for a in self._axes]}")

    def __class_getitem__(cls, item: Any) -> Type:
        return cls

    def __neg__(self) -> 'Tensor':
        """Unary negation (-x)."""
        return Tensor(-self.unwrap(), *self._axes)

    def __and__(self, other: 'Tensor') -> 'Bundle':
        """Allows bundling via the & operator: (x & y).d.proj()"""
        return Bundle(self, other)

    def item(self) -> float:
        """Extracts a scalar Tensor as a standard Python float."""
        import numpy as np
        # Convert the JAX array to a NumPy scalar, then to a Python float
        return float(np.array(self.unwrap()).item())

    def __float__(self) -> float:
        """Allows float(loss)"""
        return self.item()

    def __format__(self, format_spec: str) -> str:
        """Allows f-string formatting: f'{loss:.4f}'"""
        return format(self.item(), format_spec)

    def __dir__(self):
        """Exposes dynamic NN functions to dynamic autocomplete (Jupyter/REPL)."""
        from . import nn
        base_dir = super().__dir__()

        # Add all available axes
        axes = [a.name for a in (self.tensor.topology if hasattr(self, 'tensor') else self._axes)]

        # Add all public callable functions from your nn library!
        nn_funcs = [k for k, v in vars(nn).items() if callable(v) and not k.startswith('_')]

        return sorted(set(base_dir + axes + nn_funcs))


def wrap(raw_tensor: Any, *axes: Axis) -> Tensor:
    return Tensor(raw_tensor, *axes)


# ==========================================
# THE CONTEXT ROUTING STACK (Gather / Route)
# ==========================================

class RoutedContext:
    """The active context when a targeted axis is sliced by an index tensor."""

    def __init__(self, tensor: Tensor, target_axes: Tuple[Axis, ...], indices: Tensor):
        if len(target_axes) != 1:
            raise ValueError("Routing requires exactly one target axis.")
        self.tensor = tensor
        self.target_ax = target_axes[0]
        self.indices = indices

    def gather(self) -> Tensor:
        """
        Performs a topological gather.
        Replaces the targeted axis with the entire topology of the indices.
        """
        import jax.numpy as jnp

        raw_x = self.tensor.unwrap()
        raw_idx = self.indices.unwrap()

        if self.target_ax not in self.tensor.topology:
            raise ValueError(f"Target axis '{self.target_ax.name}' not in tensor topology.")

        ax_idx = self.tensor.topology.index(self.target_ax)

        # JAX take natively handles the underlying array lookup
        res_raw = jnp.take(raw_x, raw_idx, axis=ax_idx)

        # Compute the new topology: target axis is replaced by the index topology
        new_top = list(self.tensor.topology)
        new_top = new_top[:ax_idx] + list(self.indices.topology) + new_top[ax_idx + 1:]

        # Return a pure Tensor directly!
        return Tensor(res_raw, *new_top)


def decay_monads(x):
    """
    Converts lazy slices into plain Tensor chunks before crossing hard boundaries
    like jax.jit.

    This implements:

        slice used as value => Tensor
        slice closed with [:] => patch commit
    """
    if hasattr(x, "chunk_tensor"):
        return x.chunk_tensor

    if isinstance(x, Bundle):
        return Bundle(*[decay_monads(t) for t in x.tensors])

    if isinstance(x, tuple):
        return tuple(decay_monads(v) for v in x)

    if isinstance(x, list):
        return [decay_monads(v) for v in x]

    if isinstance(x, dict):
        return {k: decay_monads(v) for k, v in x.items()}

    return x


import jax

# Register Tensor as a native JAX PyTree!
# This is the secret sauce that allows `jax.jit` to accept and return Tensors natively.
jax.tree_util.register_pytree_node(
    Tensor,
    lambda t: ((t.unwrap(),), (t.topology,)),
    lambda aux, children: Tensor(children[0], *aux[0])
)