import inspect
from collections import defaultdict
from typing import Any, Callable
import jax


class AxiomStateManager:
    def __init__(self, seed: int = 42):
        self.params = {}
        self.counters = defaultdict(int)
        self.root_key = jax.random.PRNGKey(seed)

    def next_key(self):
        """Splits the key and securely updates the global entropy state."""
        new_key, subkey = jax.random.split(self.root_key)

        # Tracer Leak Prevention
        # We only update the global key if we are NOT inside a compiled JAX trace.
        # This prevents XLA from baking a stale key into the compiled graph.
        if not isinstance(new_key, jax.core.Tracer):
            self.root_key = new_key

        return subkey

    def _get_caller_scope(self) -> str:
        """Magically finds the user's function name by inspecting the trace stack."""
        stack = inspect.stack()
        for frame_info in stack[2:]:
            if "axiom/" not in frame_info.filename.replace("\\", "/"):
                return frame_info.function
        return "global"

    def get_param(self, layer_type: str, shape: tuple, init_fn: Callable, tie: str = None, **init_kwargs) -> Any:
        """Resolves the namespace path and retrieves or initializes the parameter."""
        if tie is not None and tie.startswith("@"):
            true_name = tie[1:]
        else:
            scope = self._get_caller_scope()
            if tie is not None:
                true_name = f"{scope}/{tie}"
            else:
                self.counters[f"{scope}/{layer_type}"] += 1
                true_name = f"{scope}/{layer_type}_{self.counters[f'{scope}/{layer_type}']}"

        if true_name in self.params:
            if self.params[true_name].shape != shape:
                raise ValueError(
                    f"Shape mismatch for tied weight '{true_name}'. Expected {shape}, got {self.params[true_name].shape}.")
            return self.params[true_name]
        else:
            # Initialize with our secure key manager!
            new_param = init_fn(self.next_key(), shape, **init_kwargs)
            self.params[true_name] = new_param
            return new_param


# The Singleton
state = AxiomStateManager()