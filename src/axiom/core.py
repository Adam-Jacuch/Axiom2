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
    """The Unslice Monad. Tracks sliced chunks and stitches them back together via [:]"""

    def __init__(self, original_tensor, target_ax, slice_obj, chunk_tensor):
        self.original_tensor = original_tensor
        self.target_ax = target_ax
        self.slice_obj = slice_obj
        self.chunk_tensor = chunk_tensor

    def pw(self, func, **kwargs) -> 'SlicedMonad':
        """Applies a pointwise function to the chunk and keeps the monad alive."""
        new_chunk = TargetedTensor(self.chunk_tensor, (self.target_ax,)).pw(func, **kwargs)
        return SlicedMonad(self.original_tensor, self.target_ax, self.slice_obj, new_chunk)

    def proj(self, *target_axes, bias: bool = False, tie: Optional[str] = None) -> 'SlicedMonad':
        """Projects the chunk and keeps the monad alive."""
        new_chunk = TargetedTensor(self.chunk_tensor, (self.target_ax,)).proj(*target_axes, bias=bias, tie=tie)
        return SlicedMonad(self.original_tensor, self.target_ax, self.slice_obj, new_chunk)

    def __mul__(self, other):
        """SwiGLU Magic: Cross-tensor math drops the stitch and returns pure Tensors!"""
        if hasattr(other, 'chunk_tensor'):
            other = other.chunk_tensor
        elif hasattr(other, 'tensor'):
            other = other.tensor

        if isinstance(other, Tensor):
            return self.chunk_tensor * other
        return SlicedMonad(self.original_tensor, self.target_ax, self.slice_obj, self.chunk_tensor * other)

    def __getitem__(self, key) -> 'Tensor':
        """The Unslice Closer [:] - Concatenates the chunk back into the original topology!"""
        if key == slice(None):
            import jax.numpy as jnp
            orig_raw = self.original_tensor.unwrap()
            ax_idx = self.original_tensor.topology.index(self.target_ax)

            # Resolve slice bounds
            start = self.slice_obj.start or 0
            stop = self.slice_obj.stop or orig_raw.shape[ax_idx]

            # Slice out the untouched left and right segments
            left_slice = [slice(None)] * orig_raw.ndim
            left_slice[ax_idx] = slice(None, start)

            right_slice = [slice(None)] * orig_raw.ndim
            right_slice[ax_idx] = slice(stop, None)

            left_raw = orig_raw[tuple(left_slice)]
            right_raw = orig_raw[tuple(right_slice)]

            # Stitch the modified chunk back into the middle!
            stitched_raw = jnp.concatenate([left_raw, self.chunk_tensor.unwrap(), right_raw], axis=ax_idx)

            # Compute the new global topology (handles if the chunk was projected to a new size/name)
            chunk_ax = self.chunk_tensor.topology[ax_idx]
            new_size = left_raw.shape[ax_idx] + chunk_ax.size + right_raw.shape[ax_idx]
            new_axis = Axis(chunk_ax.name, new_size)

            new_top = list(self.original_tensor.topology)
            new_top[ax_idx] = new_axis

            return Tensor(stitched_raw, *new_top)

        raise ValueError("Use [:] to stitch the sliced monad back into the parent tensor.")


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

    def proj(self, *target_axes: Axis, bias: bool = False, tie: Optional[str] = None) -> 'Tensor':
        if not target_axes:
            target_axes = self.target_axes

        import jax, jax.numpy as jnp, numpy as np
        from .init import _next_key

        in_dim = np.prod([a.size for a in self.target_axes])
        out_dim = np.prod([a.size for a in target_axes])
        if in_dim is None or out_dim is None:
            raise ValueError("Projections require axes to have statically defined sizes.")

        # Use the Tracer-Safe Key Tracker!
        key = _next_key()

        # Xavier/Glorot Normal Initialization (prevents exploding variance)
        variance_scale = jnp.sqrt(2.0 / (in_dim + out_dim))
        W_raw = jax.random.normal(key, (in_dim, out_dim)) * variance_scale

        tie_obj = Tie(tie) if tie else None
        W_param = Tensor(W_raw, Axis("_in", in_dim), Axis("_out", out_dim)).param(tie=tie_obj)

        # --- THE FIX: Topologically-Aware Flattening ---
        # 1. Identify which axes we are keeping vs targeting
        kept_axes = tuple(a for a in self.tensor.topology if a not in self.target_axes)

        # 2. Transpose the targeted axes to the very end of the array
        transpose_order = [self.tensor.topology.index(a) for a in kept_axes + self.target_axes]
        transposed_raw = jnp.transpose(self.tensor.unwrap(), transpose_order)

        # 3. Flatten all the targeted axes into a single 'in_dim' dimension
        kept_shape = tuple(a.size if a.size is not None else -1 for a in kept_axes)
        flattened_raw = jnp.reshape(transposed_raw, kept_shape + (in_dim,))

        # 4. Safely perform the matrix multiplication!
        result_raw = jnp.dot(flattened_raw, W_param.unwrap())

        # 5. Reconstruct the new topology and shape
        new_topology = kept_axes + target_axes
        new_shape = tuple(a.size if a.size is not None else -1 for a in new_topology)

        result_raw = jnp.reshape(result_raw, new_shape)
        result_tensor = Tensor(result_raw, *new_topology)

        # --- BIAS LOGIC ---
        if bias:
            b_raw = jnp.zeros(out_dim)
            b_tie = Tie(f"{tie}_bias") if tie else None
            b_param = Tensor(b_raw, *target_axes).param(tie=b_tie)
            return result_tensor + b_param  # Native broadcasting handles alignment!

        return result_tensor

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
        if key == slice(None): return self.tensor
        if hasattr(key, 'topology'): return RoutedContext(self.tensor, self.target_axes, key)

        if len(self.target_axes) != 1:
            raise ValueError("Slicing requires exactly one target axis.")

        target = self.target_axes[0]
        ax_idx = self.tensor.topology.index(target)

        full_slice = [slice(None)] * len(self.tensor.topology)
        full_slice[ax_idx] = key
        sliced_raw = self.tensor.unwrap()[tuple(full_slice)]

        new_size = sliced_raw.shape[ax_idx]
        new_axis = Axis(target.name, new_size)

        new_topology = list(self.tensor.topology)
        new_topology[ax_idx] = new_axis
        chunk_tensor = Tensor(sliced_raw, *new_topology)

        # TRIGGER THE MONAD!
        return SlicedMonad(self.tensor, target, key, chunk_tensor)

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

    def proj(self, *target_axes: Axis, bias: bool = False, tie: Optional[str] = None) -> Tuple['Tensor', ...]:
        """Parallel Tuple Projection. Returns a tuple of projected Tensors."""
        # For eager execution, we simply map the projection across the bundle.
        # (Later, @axiom_jit will fuse this into a block-matrix mult!)
        results = []
        for tensor in self.bundle.tensors:
            # Reconstruct a targeted tensor and call proj
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.proj(*target_axes, bias=bias, tie=tie))
        return tuple(results)

    def pw(self, func, tie: Optional[str] = None, **kwargs) -> Tuple['Tensor', ...]:
        """Parallel Pointwise Mapping."""
        results = []
        for tensor in self.bundle.tensors:
            t_tensor = TargetedTensor(tensor, self.target_axes)
            results.append(t_tensor.pw(func, tie=tie, **kwargs))
        return tuple(results)

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


import jax

# Register Tensor as a native JAX PyTree!
# This is the secret sauce that allows `jax.jit` to accept and return Tensors natively.
jax.tree_util.register_pytree_node(
    Tensor,
    lambda t: ((t.unwrap(),), (t.topology,)),
    lambda aux, children: Tensor(children[0], *aux[0])
)