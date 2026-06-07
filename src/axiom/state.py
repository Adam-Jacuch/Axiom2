from typing import Any, Callable
import jax
from .core import compiler_state


class AxiomStateManager:
    def __init__(self, seed: int = 42):
        self.root_key = jax.random.PRNGKey(seed)

    def next_key(self):
        new_key, subkey = jax.random.split(self.root_key)
        if not isinstance(new_key, jax.core.Tracer):
            self.root_key = new_key
        return subkey

    def get_param(self, layer_type: str, shape: tuple, init_fn: Callable, tie: str = None, **init_kwargs) -> Any:
        true_name = compiler_state.get_scoped_name(explicit_name=tie, fallback_prefix=layer_type)

        if true_name not in compiler_state.params:
            compiler_state.params[true_name] = init_fn(self.next_key(), shape, **init_kwargs)

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


state = AxiomStateManager()