from typing import Any, Callable
import jax
from .core import compiler_state
from .layout import ParameterLayout


class AxiomStateManager:
    def __init__(self, seed: int = 42):
        self.root_key = jax.random.PRNGKey(seed)

    def next_key(self):
        new_key, subkey = jax.random.split(self.root_key)
        if not isinstance(new_key, jax.core.Tracer):
            self.root_key = new_key
        return subkey

    def get_param(
        self,
        layer_type: str,
        shape: tuple,
        init_fn: Callable,
        tie: str = None,
        *,
        layout: ParameterLayout | None = None,
        **init_kwargs,
    ) -> Any:
        true_name = compiler_state.get_scoped_name(explicit_name=tie, fallback_prefix=layer_type)

        if true_name not in compiler_state.params:
            if compiler_state.strict_params:
                raise RuntimeError(
                    f"Axiom parameter '{true_name}' is missing from an initialized model/exported parameter "
                    "dictionary. Reinitialize the model or pass the complete parameter tree."
                )
            compiler_state.params[true_name] = init_fn(self.next_key(), shape, **init_kwargs)
        if layout is not None:
            existing_layout = compiler_state.param_layouts.get(true_name)
            if existing_layout is None:
                compiler_state.param_layouts[true_name] = layout
            elif not _layouts_compatible(
                existing_layout,
                layout,
                transpose=(compiler_state.params[true_name].shape != shape and compiler_state.params[true_name].shape == shape[::-1]),
            ):
                raise ValueError(
                    f"Tied parameter '{true_name}' was requested with an incompatible logical layout. "
                    "Use matching axis placement or introduce an explicit resharding boundary."
                )

        param = compiler_state.params[true_name]

        if hasattr(param, 'shape'):
            if param.shape == shape:
                pass  # Exact match, return normally
            elif param.shape == shape[::-1]:
                # Transpose match! (e.g., tying a (1024, 128) embed to a (128, 1024) proj)
                param = param.T
            else:
                raise ValueError(
                    f"Shape mismatch for tied weight '{true_name}'. Expected {shape} or its transpose {shape[::-1]}, got {param.shape}.")

        return param


def _axis_signature(axis):
    return (
        axis.name,
        axis.size,
        getattr(getattr(axis, "placement", None), "name", None),
        getattr(axis, "replicated", False),
    )


def _layouts_compatible(existing: ParameterLayout, requested: ParameterLayout, *, transpose: bool) -> bool:
    if existing.metadata() == requested.metadata():
        return True
    if not transpose or existing.kind != "projection" or requested.kind != "projection":
        return False
    return (
        tuple(_axis_signature(axis) for axis in existing.input_axes)
        == tuple(_axis_signature(axis) for axis in requested.output_axes)
        and tuple(_axis_signature(axis) for axis in existing.output_axes)
        == tuple(_axis_signature(axis) for axis in requested.input_axes)
    )


state = AxiomStateManager()
