import inspect
from typing import Any, Callable
import jax

# Import the global execution context
from .core import compiler_state


class AxiomStateManager:
    def __init__(self, seed: int = 42):
        # We no longer need self.params or self.counters!
        # The Ghost Pass and XLA Trace manage all of that seamlessly via compiler_state.
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

        # 1. Deterministic Naming
        # We blend your dynamic scoping with the compiler's resettable counter!
        if tie is not None and tie.startswith("@"):
            true_name = tie[1:]
        else:
            scope = self._get_caller_scope()
            if tie is not None:
                true_name = f"{scope}/{tie}"
            else:
                true_name = f"{scope}/{layer_type}_{compiler_state.param_counter}"
                # Increment the global, trace-safe counter!
                compiler_state.param_counter += 1

        # 2. Ghost Pass Allocation
        # Write DIRECTLY to compiler_state so the PyTree can package it up!
        if getattr(compiler_state, 'is_initializing', False):
            if true_name not in compiler_state.params:
                # Initialize with our secure key manager!
                new_param = init_fn(self.next_key(), shape, **init_kwargs)
                compiler_state.params[true_name] = new_param

        # 3. Execution & Validation
        # Read the active Tracers seamlessly from compiler_state.params
        param = compiler_state.params[true_name]

        # Safe shape check (Some JAX tracers hide their shape under .aval, but standard arrays have .shape)
        if hasattr(param, 'shape') and param.shape != shape:
            raise ValueError(
                f"Shape mismatch for tied weight '{true_name}'. "
                f"Expected {shape}, got {param.shape}."
            )

        return param


# The Singleton
state = AxiomStateManager()