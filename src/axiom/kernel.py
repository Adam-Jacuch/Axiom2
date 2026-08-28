"""Named-axis lowering for small Pallas kernels.

The public surface deliberately describes *logical* tiles.  This module owns
the translation to Pallas grids and ``BlockSpec`` objects, so callers never
need to hand-write positional block mappings for ordinary tiled maps.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl


@dataclass(frozen=True)
class _KernelContext:
    """The named grid/tile values visible while a tile function is traced."""

    program_ids: Mapping[str, Any]
    tile_sizes: Mapping[str, int]


_KERNEL_CONTEXT: ContextVar[_KernelContext | None] = ContextVar(
    "axiom_kernel_context", default=None
)


@contextmanager
def kernel_context(program_ids: Mapping[str, Any], tile_sizes: Mapping[str, int]):
    """Make Pallas grid information available through ``ax.grid`` and ``ax.tile``."""
    token = _KERNEL_CONTEXT.set(_KernelContext(program_ids, tile_sizes))
    try:
        yield
    finally:
        _KERNEL_CONTEXT.reset(token)


class _GridNamespace:
    def __getitem__(self, axis):
        context = _KERNEL_CONTEXT.get()
        if context is None:
            raise RuntimeError("ax.grid[...] is only available inside a tiled .map() body.")
        name = _axis_name(axis)
        try:
            return context.program_ids[name]
        except KeyError as exc:
            raise ValueError(f"Axis '{name}' is not part of this map grid.") from exc


class _TileNamespace:
    def __getitem__(self, axis):
        context = _KERNEL_CONTEXT.get()
        if context is None:
            raise RuntimeError("ax.tile[...] is only available inside a tiled .map() body.")
        name = _axis_name(axis)
        try:
            return jnp.arange(context.tile_sizes[name], dtype=jnp.int32)
        except KeyError as exc:
            raise ValueError(f"Axis '{name}' is not tiled by this map.") from exc


grid = _GridNamespace()
tile = _TileNamespace()


def _axis_name(axis: Any) -> str:
    name = getattr(axis, "name", None)
    if not isinstance(name, str):
        raise TypeError("Expected an Axiom Axis, e.g. ax.s, for a tiled axis lookup.")
    return name


def _source_tensors(source) -> tuple:
    from .core import Bundle, Tensor

    if isinstance(source, Tensor):
        return (source,)
    if isinstance(source, Bundle):
        return tuple(source.tensors)
    raise TypeError("Tiled kernels can only map Axiom Tensor or Bundle inputs.")


def _rebuild_source(original, tensors: Sequence):
    from .core import Bundle

    return tensors[0] if len(tensors) == 1 and not isinstance(original, Bundle) else Bundle(*tensors)


def _result_tensors(result) -> tuple:
    return _source_tensors(result)


def _result_kind(result) -> str:
    from .core import Bundle, Tensor

    if isinstance(result, Tensor):
        return "tensor"
    if isinstance(result, Bundle):
        return "bundle"
    raise TypeError("A tiled map function must return an Axiom Tensor or Bundle.")


class _PendingAxis:
    """A callable placeholder returned by ``stream.axis`` before its tile is given."""

    def __init__(self, stream: "TiledAxisRef", axis_name: str):
        self._stream = stream
        self._axis_name = axis_name

    def __call__(self, tile_size: int) -> "AxisStream":
        return self._stream._with_axis(self._axis_name, tile_size)


class TiledAxisRef:
    """An immutable named tile plan for one Tensor or parallel Bundle.

    ``tensor.s(128)`` produces a ``TiledAxisRef``.  Chaining another named
    axis extends the plan without exposing positional Pallas grid dimensions.
    """

    def __init__(self, source, tiles: Mapping[str, int]):
        self._source = source
        self._tiles = OrderedDict(tiles)
        self._validate_tiles()

    @property
    def source(self):
        return self._source

    @property
    def tiles(self) -> Mapping[str, int]:
        return self._tiles.copy()

    def _validate_tiles(self) -> None:
        tensors = _source_tensors(self._source)
        for name, tile_size in self._tiles.items():
            if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
                raise ValueError(f"Tile size for axis '{name}' must be a positive integer, got {tile_size!r}.")

            matching_axes = [a for t in tensors for a in t.topology if a.name == name]
            if not matching_axes:
                raise ValueError(f"Axis '{name}' does not exist on this tiled input.")
            if any(a.size is None for a in matching_axes):
                raise ValueError(f"Axis '{name}' must have a static size before it can be tiled.")
            sizes = {a.size for a in matching_axes}
            if len(sizes) != 1:
                raise ValueError(f"Bundle axis '{name}' has inconsistent sizes: {sorted(sizes)}.")
            axis_size = next(iter(sizes))

    def _with_axis(self, axis_name: str, tile_size: int) -> "AxisStream":
        if axis_name in self._tiles:
            raise ValueError(f"Axis '{axis_name}' already has tile size {self._tiles[axis_name]}.")
        tiles = OrderedDict(self._tiles)
        tiles[axis_name] = tile_size
        return AxisStream(self._source, tiles)

    def __getattr__(self, name: str) -> _PendingAxis:
        if any(axis.name == name for tensor in _source_tensors(self._source) for axis in tensor.topology):
            return _PendingAxis(self, name)
        raise AttributeError(f"Axis '{name}' is not present on this tiled input.")

    def map(
        self,
        tile_fn: Callable,
        *,
        interpret: bool | None = None,
        compiler_params: Any = None,
        name: str | None = None,
    ):
        """Run ``tile_fn`` once per logical grid position through ``pallas_call``.

        Inputs absent from a requested grid axis are replicated for that axis.
        Outputs must retain only source axes and must be owned by every grid
        axis they omit; otherwise multiple programs would write the same HBM
        block and the map is rejected.
        """
        if not callable(tile_fn):
            raise TypeError("map() expects a callable tile function.")
        if not self._tiles:
            raise ValueError("map() requires at least one tiled axis, e.g. tensor.s(128).map(fn).")

        from .core import Axis, Bundle, Tensor

        input_tensors = _source_tensors(self._source)
        grid_axes = tuple(self._tiles)
        grid_axis_positions = {axis_name: i for i, axis_name in enumerate(grid_axes)}
        canonical_axes = _canonical_axes(input_tensors)
        grid_shape = tuple(
            _ceil_div(canonical_axes[name].size, self._tiles[name]) for name in grid_axes
        )
        local_topologies = tuple(
            tuple(Axis(axis.name, self._tiles.get(axis.name, axis.size)) for axis in tensor.topology)
            for tensor in input_tensors
        )

        # The abstract pass both validates the public function and records its
        # named output topology.  ``ax.grid`` intentionally reads zero here;
        # its only contract at this phase is scalar shape/dtype information.
        captured: dict[str, Any] = {}

        def abstract_tile_fn(*local_inputs):
            local_tensors = tuple(
                Tensor(raw, *topology) for raw, topology in zip(local_inputs, local_topologies)
            )
            with kernel_context(
                {axis_name: jnp.asarray(0, dtype=jnp.int32) for axis_name in grid_axes}, self._tiles
            ):
                result = tile_fn(_rebuild_source(self._source, local_tensors))
            kind = _result_kind(result)
            result_tensors = _result_tensors(result)
            captured["kind"] = kind
            captured["topologies"] = tuple(t.topology for t in result_tensors)
            return tuple(t.unwrap() for t in result_tensors)

        abstract_inputs = tuple(
            jax.ShapeDtypeStruct(tuple(axis.size for axis in topology), t.unwrap().dtype)
            for t, topology in zip(input_tensors, local_topologies)
        )
        output_avals = jax.eval_shape(abstract_tile_fn, *abstract_inputs)
        output_topologies = captured.get("topologies")
        if output_topologies is None:
            raise RuntimeError("Unable to infer tiled map output topology.")
        if not isinstance(output_avals, tuple):
            output_avals = (output_avals,)

        output_global_topologies = tuple(
            _global_output_topology(topology, canonical_axes, self._tiles)
            for topology in output_topologies
        )
        _validate_output_ownership(output_global_topologies, grid_axes)

        in_specs = tuple(
            _block_spec(tensor.topology, self._tiles, grid_axis_positions) for tensor in input_tensors
        )
        out_specs = tuple(
            _block_spec(topology, self._tiles, grid_axis_positions)
            for topology in output_global_topologies
        )
        out_shape = tuple(
            jax.ShapeDtypeStruct(tuple(axis.size for axis in topology), aval.dtype)
            for aval, topology in zip(output_avals, output_global_topologies)
        )

        def pallas_kernel(*refs):
            input_refs = refs[: len(input_tensors)]
            output_refs = refs[len(input_tensors):]
            local_tensors = tuple(
                Tensor(
                    _zero_pad_tail(ref[...], topology, tensor.topology, self._tiles, grid_axis_positions),
                    *topology,
                )
                for ref, tensor, topology in zip(input_refs, input_tensors, local_topologies)
            )
            program_ids = {
                axis_name: pl.program_id(position)
                for axis_name, position in grid_axis_positions.items()
            }
            with kernel_context(program_ids, self._tiles):
                result = tile_fn(_rebuild_source(self._source, local_tensors))
            result_tensors = _result_tensors(result)
            if _result_kind(result) != captured["kind"] or len(result_tensors) != len(output_refs):
                raise TypeError("A tiled map function must return a stable Tensor/Bundle structure.")
            for output_ref, output_tensor in zip(output_refs, result_tensors):
                output_ref[...] = output_tensor.unwrap()

        if interpret is None:
            interpret = jax.devices()[0].platform == "cpu"

        mapped = pl.pallas_call(
            pallas_kernel,
            out_shape=out_shape,
            grid=grid_shape,
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=interpret,
            compiler_params=compiler_params,
            name=name,
        )
        # Current Pallas releases do not implement reverse-mode autodiff for
        # pallas_call.  Keep the hardware kernel as the forward path and give
        # JAX a mathematically identical, named-tile expansion for the VJP.
        # The expansion is only traced for differentiation; ordinary execution
        # still calls the Pallas kernel above.
        def eager_tiled(*raw_inputs):
            output_buffers = [jnp.zeros(shape.shape, shape.dtype) for shape in out_shape]
            for grid_indices in product(*(range(size) for size in grid_shape)):
                local_tensors = []
                for raw, tensor, topology in zip(raw_inputs, input_tensors, local_topologies):
                    slices = tuple(
                        slice(
                            grid_indices[grid_axis_positions[axis.name]] * self._tiles[axis.name],
                            (grid_indices[grid_axis_positions[axis.name]] + 1) * self._tiles[axis.name],
                        )
                        if axis.name in self._tiles
                        else slice(None)
                        for axis in tensor.topology
                    )
                    local_raw = raw[slices]
                    padding = tuple(
                        (0, local_axis.size - local_raw.shape[dimension])
                        for dimension, local_axis in enumerate(topology)
                    )
                    local_tensors.append(Tensor(jnp.pad(local_raw, padding), *topology))
                with kernel_context(
                    {
                        axis_name: jnp.asarray(grid_indices[position], dtype=jnp.int32)
                        for axis_name, position in grid_axis_positions.items()
                    },
                    self._tiles,
                ):
                    eager_result = tile_fn(_rebuild_source(self._source, local_tensors))
                eager_tensors = _result_tensors(eager_result)
                for output_index, (output, topology) in enumerate(
                    zip(eager_tensors, output_global_topologies)
                ):
                    slices = tuple(
                        slice(
                            grid_indices[grid_axis_positions[axis.name]] * self._tiles[axis.name],
                            (grid_indices[grid_axis_positions[axis.name]] + 1) * self._tiles[axis.name],
                        )
                        if axis.name in self._tiles
                        else slice(None)
                        for axis in topology
                    )
                    valid_shape = tuple(
                        stop - start
                        for start, stop in (
                            _slice_bounds(axis, grid_indices, grid_axis_positions, self._tiles)
                            for axis in topology
                        )
                    )
                    valid_slices = tuple(slice(0, size) for size in valid_shape)
                    output_buffers[output_index] = output_buffers[output_index].at[slices].set(
                        output.unwrap()[valid_slices]
                    )
            return tuple(output_buffers)

        @jax.custom_vjp
        def execute(*raw_inputs):
            result = mapped(*raw_inputs)
            return result if isinstance(result, tuple) else (result,)

        def execute_fwd(*raw_inputs):
            return execute(*raw_inputs), raw_inputs

        def execute_bwd(raw_inputs, output_cotangents):
            _, pullback = jax.vjp(eager_tiled, *raw_inputs)
            return pullback(output_cotangents)

        execute.defvjp(execute_fwd, execute_bwd)
        raw_results = execute(*(tensor.unwrap() for tensor in input_tensors))
        if not isinstance(raw_results, tuple):
            raw_results = (raw_results,)
        result_tensors = tuple(
            Tensor(raw, *topology) for raw, topology in zip(raw_results, output_global_topologies)
        )
        return result_tensors[0] if captured["kind"] == "tensor" else Bundle(*result_tensors)

    def fold(
        self,
        step_fn: Callable,
        *,
        init: Any,
        until: Any = None,
        stages: int = 1,
    ):
        """Fold fixed-size subtiles in registers with ``lax.fori_loop``.

        The source is expected to be a tile-local value when used from a
        ``map`` body.  No intermediate fold carry is materialized as an Axiom
        output.  ``until`` is an exclusive, tile-count bound and may be a JAX
        scalar (including a value derived from ``ax.grid``).  ``stages`` is a
        validated scheduling hint: it preserves semantics on all backends;
        backend-specific asynchronous pipeline lowering can consume it later.
        """
        if not callable(step_fn):
            raise TypeError("fold() expects a callable step function.")
        if not isinstance(stages, int) or isinstance(stages, bool) or stages < 1:
            raise ValueError("stages must be a positive integer.")
        if len(self._tiles) != 1:
            raise ValueError("fold() currently operates on one tiled axis; create a single-axis stream.")

        from .core import Axis, Bundle, Tensor

        (axis_name, tile_size), = self._tiles.items()
        source_tensors = _source_tensors(self._source)
        axis_positions = []
        for tensor in source_tensors:
            try:
                axis_positions.append(next(i for i, axis in enumerate(tensor.topology) if axis.name == axis_name))
            except StopIteration as exc:
                raise ValueError(f"Fold axis '{axis_name}' is missing from an input tensor.") from exc

        full_size = next(axis.size for axis in source_tensors[0].topology if axis.name == axis_name)
        num_tiles = _ceil_div(full_size, tile_size)
        limit = jnp.asarray(num_tiles if until is None else until, dtype=jnp.int32)
        limit = jnp.clip(limit, 0, num_tiles)

        def local_step(carry, index):
            local_tensors = []
            for tensor, axis_position in zip(source_tensors, axis_positions):
                padding = [(0, 0)] * tensor.unwrap().ndim
                padding[axis_position] = (0, num_tiles * tile_size - full_size)
                padded = jnp.pad(tensor.unwrap(), tuple(padding))
                raw = lax.dynamic_slice_in_dim(padded, index * tile_size, tile_size, axis=axis_position)
                topology = list(tensor.topology)
                topology[axis_position] = Axis(axis_name, tile_size)
                local_tensors.append(Tensor(raw, *topology))
            step_input = _rebuild_source(self._source, local_tensors)
            return step_fn(carry, step_input)

        def body(index, carry):
            return lax.cond(index < limit, lambda value: local_step(value, index), lambda value: value, carry)

        # ``stages`` intentionally has no arithmetic effect: generic Pallas
        # pallas_call has no portable async-buffer API.  Keeping it explicit
        # makes the future TPU/Mosaic pipeline lowering source-compatible.
        return lax.fori_loop(0, num_tiles, body, init)


class AxisStream(TiledAxisRef):
    """A multi-axis ``TiledAxisRef``; kept as a public name for inspection."""


def tile_axis(source, axis_name: str, tile_size: int) -> AxisStream:
    """Create a first named tile in response to ``tensor.axis(tile_size)``."""
    return AxisStream(source, OrderedDict(((axis_name, tile_size),)))


def _canonical_axes(tensors: Sequence) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for tensor in tensors:
        for axis in tensor.topology:
            previous = axes.get(axis.name)
            if previous is not None and previous.size != axis.size:
                raise ValueError(f"Axis '{axis.name}' has inconsistent sizes across tiled inputs.")
            axes[axis.name] = axis
    return axes


def _global_output_topology(local_topology, canonical_axes: Mapping[str, Any], tiles: Mapping[str, int]):
    from .core import Axis

    global_axes = []
    for axis in local_topology:
        canonical = canonical_axes.get(axis.name)
        if canonical is None:
            raise ValueError(
                f"Tiled map output axis '{axis.name}' does not belong to an input. "
                "Output-axis creation is not supported by automatic BlockSpec inference."
            )
        expected_local_size = tiles.get(axis.name, canonical.size)
        if axis.size != expected_local_size:
            raise ValueError(
                f"Output axis '{axis.name}' has local size {axis.size}; expected {expected_local_size}. "
                "A tiled map must preserve its declared block geometry."
            )
        global_axes.append(Axis(axis.name, canonical.size))
    return tuple(global_axes)


def _validate_output_ownership(output_topologies, grid_axes: Sequence[str]) -> None:
    for topology in output_topologies:
        names = {axis.name for axis in topology}
        omitted = [axis_name for axis_name in grid_axes if axis_name not in names]
        if omitted:
            raise ValueError(
                "Tiled map output omits grid axis/axes "
                f"{omitted}; multiple Pallas programs would write the same output block. "
                "Keep those axes in the output or perform the reduction inside one program/fold."
            )


def _block_spec(topology, tiles: Mapping[str, int], grid_axis_positions: Mapping[str, int]):
    block_shape = tuple(tiles.get(axis.name, axis.size) for axis in topology)
    if any(size is None for size in block_shape):
        raise ValueError("Pallas BlockSpec inference requires statically sized axes.")

    def index_map(*grid_indices):
        return tuple(
            grid_indices[grid_axis_positions[axis.name]] if axis.name in tiles else 0
            for axis in topology
        )

    return pl.BlockSpec(block_shape=block_shape, index_map=index_map)


def _ceil_div(dividend: int, divisor: int) -> int:
    return (dividend + divisor - 1) // divisor


def _slice_bounds(axis, grid_indices, grid_axis_positions, tiles):
    """Return the valid global slice bounds for one output dimension."""
    if axis.name not in tiles:
        return 0, axis.size
    start = grid_indices[grid_axis_positions[axis.name]] * tiles[axis.name]
    return start, min(start + tiles[axis.name], axis.size)


def _zero_pad_tail(raw, local_topology, full_topology, tiles, grid_axis_positions):
    """Replace Pallas's unspecified blocked-tail input padding with zeroes.

    Pallas discards stores beyond an output boundary but deliberately leaves
    out-of-bounds input values unspecified.  Zeroing them gives map bodies and
    the differentiable fallback identical, deterministic tail semantics.
    """
    valid = None
    rank = len(local_topology)
    for dimension, (local_axis, full_axis) in enumerate(zip(local_topology, full_topology)):
        if local_axis.name not in tiles:
            continue
        offsets = jnp.arange(local_axis.size, dtype=jnp.int32)
        starts = pl.program_id(grid_axis_positions[local_axis.name]) * tiles[local_axis.name]
        dimension_valid = (starts + offsets) < full_axis.size
        shape = [1] * rank
        shape[dimension] = local_axis.size
        dimension_valid = jnp.reshape(dimension_valid, shape)
        valid = dimension_valid if valid is None else jnp.logical_and(valid, dimension_valid)
    return raw if valid is None else jnp.where(valid, raw, jnp.zeros_like(raw))
