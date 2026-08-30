"""Explicit device-mesh layouts for Axiom's named axes.

This module deliberately contains no Tensor operations.  It turns the small
piece of information carried by an Axis ("this logical axis lives on tp")
into JAX ``NamedSharding`` objects at the compiler boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any, Iterable, Mapping, Sequence

import jax
import numpy as np
from jax.sharding import Mesh as JaxMesh
from jax.sharding import NamedSharding, PartitionSpec


_MESH_IDS = count()


@dataclass(frozen=True)
class MeshAxis:
    """A named dimension of an :class:`AxiomMesh`.

    It is intentionally not a string: ``x.d[mesh.tp]`` can therefore reject a
    token from a different mesh rather than accepting a spelling coincidence.
    """

    name: str
    mesh_id: int

    _axiom_mesh_axis = True

    def __repr__(self) -> str:
        return self.name


class AxiomMesh:
    """A validated, explicit logical mesh over a concrete set of JAX devices."""

    def __init__(self, *, devices: Sequence[Any] | None = None, **axis_sizes: int):
        if not axis_sizes:
            raise ValueError("ax.mesh() requires at least one named mesh axis, e.g. ax.mesh(dp=8, tp=8).")
        if any(not isinstance(name, str) or not name.isidentifier() for name in axis_sizes):
            raise ValueError("Mesh axis names must be valid Python identifiers.")
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in axis_sizes.values()):
            raise ValueError("Mesh axis sizes must be positive integers.")

        selected_devices = tuple(jax.devices() if devices is None else devices)
        expected = int(np.prod(tuple(axis_sizes.values())))
        if len(selected_devices) != expected:
            raise ValueError(
                f"Mesh {dict(axis_sizes)!r} requires exactly {expected} devices, but received "
                f"{len(selected_devices)}. Pass devices=... to choose a matching device set."
            )

        self._id = next(_MESH_IDS)
        self.axis_sizes = dict(axis_sizes)
        self.axis_names = tuple(axis_sizes)
        self.devices = selected_devices
        self.jax_mesh = JaxMesh(np.asarray(selected_devices, dtype=object).reshape(tuple(axis_sizes.values())), self.axis_names)
        self._axes = {name: MeshAxis(name, self._id) for name in self.axis_names}

    def __getattr__(self, name: str) -> MeshAxis:
        try:
            return self._axes[name]
        except KeyError as exc:
            raise AttributeError(f"Mesh has no axis '{name}'. Available: {self.axis_names}.") from exc

    def __repr__(self) -> str:
        sizes = ", ".join(f"{name}={size}" for name, size in self.axis_sizes.items())
        return f"AxiomMesh({sizes})"

    def validate_token(self, token: MeshAxis) -> None:
        if not isinstance(token, MeshAxis):
            raise TypeError("Axis placement must use a mesh token such as mesh.tp, not a string.")
        if token.mesh_id != self._id or token.name not in self.axis_sizes:
            raise ValueError("This mesh-axis token belongs to a different ax.mesh() instance.")

    def named_sharding(self, spec: PartitionSpec) -> NamedSharding:
        return NamedSharding(self.jax_mesh, spec)


def placement_name(axis: Any, mesh: AxiomMesh) -> str | None:
    """Return an Axis's validated mesh-axis name, or ``None`` for replication."""
    token = getattr(axis, "placement", None)
    if token is None:
        return None
    mesh.validate_token(token)
    return token.name


def partition_spec_for_axes(axes: Iterable[Any], mesh: AxiomMesh) -> PartitionSpec:
    axes = tuple(axes)
    names = tuple(placement_name(axis, mesh) for axis in axes)
    assigned = [name for name in names if name is not None]
    if len(set(assigned)) != len(assigned):
        raise ValueError(
            "A logical tensor cannot use one mesh axis for more than one physical dimension. "
            "Split/merge the logical axis differently or use a projection layout."
        )
    _validate_axis_sizes(axes, names, mesh)
    return PartitionSpec(*names)


def _validate_axis_sizes(axes: Sequence[Any], placement_names: Sequence[str | None], mesh: AxiomMesh) -> None:
    """Reject layouts that JAX cannot partition evenly before compilation."""
    for axis, placement_name in zip(axes, placement_names):
        if placement_name is None:
            continue
        size = getattr(axis, "size", None)
        mesh_size = mesh.axis_sizes[placement_name]
        if size is not None and size % mesh_size:
            raise ValueError(
                f"Axis '{getattr(axis, 'name', '?')}' has size {size}, which cannot be evenly sharded "
                f"over mesh.{placement_name}={mesh_size}. Pad, slice to a divisible size, or replicate it with [None]."
            )


@dataclass(frozen=True)
class ParameterLayout:
    """Logical provenance needed to lower a parameter to a legal PartitionSpec."""

    axes: tuple[Any, ...]
    kind: str = "tensor"
    input_axes: tuple[Any, ...] = ()
    output_axes: tuple[Any, ...] = ()

    def partition_spec(self, mesh: AxiomMesh) -> PartitionSpec:
        if self.kind != "projection":
            return partition_spec_for_axes(self.axes, mesh)

        # A linear weight cannot be sharded along the same mesh dimension on
        # both matrix dimensions.  Prefer a column layout when its result axis
        # is placed; JAX inserts the necessary collective for a sharded input.
        output_names = [placement_name(axis, mesh) for axis in self.output_axes]
        input_names = [placement_name(axis, mesh) for axis in self.input_axes]
        _validate_axis_sizes(self.output_axes, output_names, mesh)
        _validate_axis_sizes(self.input_axes, input_names, mesh)
        output_name = next((name for name in output_names if name is not None), None)
        input_name = next((name for name in input_names if name is not None), None)
        if output_name is not None:
            return PartitionSpec(None, output_name)
        if input_name is not None:
            return PartitionSpec(input_name, None)
        return PartitionSpec(None, None)

    def metadata(self) -> dict[str, Any]:
        def encode(axes):
            return [
                {
                    "name": axis.name,
                    "size": int(axis.size) if axis.size is not None else None,
                    "placement": getattr(getattr(axis, "placement", None), "name", None),
                    "replicated": getattr(axis, "replicated", False),
                }
                for axis in axes
            ]
        return {
            "kind": self.kind,
            "axes": encode(self.axes),
            "input_axes": encode(self.input_axes),
            "output_axes": encode(self.output_axes),
        }


class AxiomLayout:
    """The JAX-facing layout companion returned by ``ax.to_jax(..., sharding=True)``."""

    def __init__(self, mesh: AxiomMesh, parameter_layouts: Mapping[str, ParameterLayout]):
        self.mesh = mesh
        self.parameter_layouts = dict(parameter_layouts)

    def parameter_specs(self, params: Mapping[str, Any]) -> dict[str, PartitionSpec]:
        missing = set(params) - set(self.parameter_layouts)
        if missing:
            raise ValueError(f"No Axiom layout metadata exists for parameter(s): {sorted(missing)}.")
        return {name: self.parameter_layouts[name].partition_spec(self.mesh) for name in params}

    def parameter_shardings(self, params: Mapping[str, Any]) -> dict[str, NamedSharding]:
        return {name: self.mesh.named_sharding(spec) for name, spec in self.parameter_specs(params).items()}

    # Short aliases make the native JAX hand-off pleasant to use.
    specs = parameter_specs
    shardings = parameter_shardings

    def place_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        shardings = self.parameter_shardings(params)
        return {name: jax.device_put(value, shardings[name]) for name, value in params.items()}

    def tensor_spec(self, tensor_or_axes: Any) -> PartitionSpec:
        axes = getattr(tensor_or_axes, "topology", tensor_or_axes)
        return partition_spec_for_axes(axes, self.mesh)

    def tensor_sharding(self, tensor_or_axes: Any) -> NamedSharding:
        return self.mesh.named_sharding(self.tensor_spec(tensor_or_axes))

    input_sharding = tensor_sharding
    output_sharding = tensor_sharding

    def state_shardings(self, state: Any, params: Mapping[str, Any], *, strict: bool = False) -> Any:
        """Infer optimizer-state placement conservatively.

        Standard optimizers allocate ``zeros_like(param)`` and consequently
        already inherit placement from :meth:`place_params`.  For external
        state trees, matching parameter-shaped leaves receive that parameter's
        layout; scalars and unknown leaves are safely replicated unless
        ``strict=True`` asks for an explicit mapping.
        """
        parameter_shardings = self.parameter_shardings(params)
        candidates: dict[tuple[int, ...], list[NamedSharding]] = {}
        for name, value in params.items():
            candidates.setdefault(tuple(value.shape), []).append(parameter_shardings[name])

        def parameter_name_from_path(path):
            for entry in reversed(path):
                key = getattr(entry, "key", None)
                if isinstance(key, str) and key in parameter_shardings:
                    return key
            return None

        def infer(path, leaf):
            shape = tuple(getattr(leaf, "shape", ()))
            if not shape:
                return self.mesh.named_sharding(PartitionSpec())
            parameter_name = parameter_name_from_path(path)
            if parameter_name is not None and shape == tuple(params[parameter_name].shape):
                # Optimizers normally retain the parameter mapping under
                # slots such as ``mu``/``nu``.  Path-aware inference keeps
                # equally shaped row- and column-parallel slots distinct.
                return parameter_shardings[parameter_name]
            options = candidates.get(shape, ())
            if len(options) == 1 or (options and all(option == options[0] for option in options)):
                return options[0]
            existing = getattr(leaf, "sharding", None)
            if isinstance(existing, NamedSharding):
                return existing
            if strict:
                raise ValueError(
                    f"Cannot infer optimizer-state sharding for shape {shape}; pass a state tree with explicit "
                    "NamedSharding or use strict=False for replicated auxiliary state."
                )
            return self.mesh.named_sharding(PartitionSpec())

        return jax.tree_util.tree_map_with_path(infer, state)

    def place_state(self, state: Any, params: Mapping[str, Any], *, strict: bool = False) -> Any:
        return jax.tree_util.tree_map(jax.device_put, state, self.state_shardings(state, params, strict=strict))

    def explain(self, params: Mapping[str, Any]) -> dict[str, PartitionSpec]:
        """Return the exact native specs selected for each parameter."""
        return self.parameter_specs(params)

    def metadata(self) -> dict[str, Any]:
        return {
            "mesh_axes": {name: int(size) for name, size in self.mesh.axis_sizes.items()},
            "parameters": {name: layout.metadata() for name, layout in self.parameter_layouts.items()},
        }
