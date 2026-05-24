import jax
from jax.tree_util import register_pytree_node
from functools import wraps
from typing import Callable, Any, TypeVar, Optional, Tuple, Dict

from .core import compiler_state, decay_monads

F = TypeVar('F', bound=Callable[..., Any])

# Ensure the tracer routing attributes exist on the singleton
compiler_state.step_params = None


# ==========================================
# 1. THE PYTREE MODEL
# ==========================================
class AxiomModel:
    """A mathematically pure PyTree wrapper for Axiom functions."""

    def __init__(self, fn: Callable, params: Optional[Dict] = None):
        # Static Auxiliary Data (The execution logic)
        self.fn = fn
        self.is_initialized = params is not None

        # Dynamic PyTree Leaves (The hardware memory)
        self.params = params if params is not None else {}

    def __call__(self, *args, **kwargs):
        # Decay monads to pure Tensors before crossing the execution boundary
        args = decay_monads(args)
        kwargs = decay_monads(kwargs)

        # --- THE CONTEXT SWITCH ---
        # 1. Snapshot the current trace state
        prev_params = getattr(compiler_state, 'params', {})
        prev_counter = getattr(compiler_state, 'param_counter', 0)
        prev_frames = compiler_state.active_frames.copy()
        prev_calls = compiler_state.func_calls.copy()

        # 2. Inject this model's state
        compiler_state.params = self.params
        compiler_state.param_counter = 0
        compiler_state.active_frames.clear()
        compiler_state.func_calls.clear()

        # 3. Execute!
        is_uninitialized = not self.is_initialized
        is_global_init = getattr(compiler_state, 'is_initializing', False)

        if is_global_init or is_uninitialized:
            # --- THE GHOST PASS ---
            compiler_state.is_initializing = True
            res = self.fn(*args, **kwargs)

            # Save the newly allocated parameters back into the PyTree
            self.params = compiler_state.params.copy()
            self.is_initialized = True

            # Restore global init state
            compiler_state.is_initializing = is_global_init
        else:
            # --- PURE XLA EXECUTION ---
            res = self.fn(*args, **kwargs)

        # --- RESTORE CONTEXT ---
        compiler_state.param_counter = prev_counter
        compiler_state.active_frames = prev_frames
        compiler_state.func_calls = prev_calls
        compiler_state.params = prev_params

        return res

    # --- NATIVE MODEL CALCULUS (For Meta-Learning & RL) ---
    def __sub__(self, other):
        if isinstance(other, dict):
            new_params = jax.tree_util.tree_map(lambda p, g: p - g, self.params, other)
            return AxiomModel(self.fn, new_params)
        raise TypeError("Can only subtract parameter dictionaries from an AxiomModel.")

    def __add__(self, other):
        # Allow adding two Neural Networks together!
        if isinstance(other, AxiomModel):
            new_params = jax.tree_util.tree_map(lambda p1, p2: p1 + p2, self.params, other.params)
            return AxiomModel(self.fn, new_params)
        # Allow adding gradient dictionaries
        if isinstance(other, dict):
            new_params = jax.tree_util.tree_map(lambda p, g: p + g, self.params, other)
            return AxiomModel(self.fn, new_params)
        raise TypeError("Can only add parameter dicts or AxiomModels to an AxiomModel.")

    def __mul__(self, scalar: float):
        # Allow multiplying an entire Neural Network by a scalar!
        if isinstance(scalar, (int, float)):
            new_params = jax.tree_util.tree_map(lambda p: p * scalar, self.params)
            return AxiomModel(self.fn, new_params)
        raise TypeError("Can only multiply an AxiomModel by a scalar float/int.")

    # --- DICTIONARY DUCK-TYPING (For ergonomic gradient routing) ---
    def items(self):
        return self.params.items()

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def __getitem__(self, key: str):
        return self.params[key]

    def __setitem__(self, key: str, value: Any):
        self.params[key] = value

    def __contains__(self, key: str):
        return key in self.params

    # ADD THESE TWO METHODS:
    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)


# Tell JAX how to flatten and unflatten our model!
def _unflatten_model(aux, children):
    fn = aux[0]  # ONLY the function is static aux data now!
    params = children[0]

    model = AxiomModel(fn, params)
    # If JAX is unflattening this, it means we are inside the trace or execution.
    # Therefore, the Ghost Pass has already guaranteed initialization!
    model.is_initialized = True
    return model


register_pytree_node(
    AxiomModel,
    lambda m: ((m.params,), (m.fn,)),  # Flatten: Removed m.is_initialized!
    _unflatten_model  # Unflatten
)


# ==========================================
# 2. GHOST INITIALIZATION WRAPPERS
# ==========================================
def _trigger_ghost_pass(fn, *args, **kwargs):
    """Eagerly runs the function to auto-initialize any uninitialized AxiomModels."""

    # Custom recursive walker to find AxiomModels WITHOUT flattening them into leaves!
    def _get_models(obj):
        models = []
        if isinstance(obj, AxiomModel):
            models.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                models.extend(_get_models(item))
        elif isinstance(obj, dict):
            for item in obj.values():
                models.extend(_get_models(item))
        return models

    models = _get_models(args) + _get_models(kwargs)

    if any(not m.is_initialized for m in models):
        compiler_state.is_initializing = True
        compiler_state.reset_pass_state()
        fn(*args, **kwargs)
        compiler_state.is_initializing = False


class AxiomJitWrapper:
    def __init__(self, fn):
        self.fn = fn
        self._jitted_fn = None

    def __call__(self, *args, **kwargs):
        # 1. Trigger the Ghost Pass if any model is uninitialized
        _trigger_ghost_pass(self.fn, *args, **kwargs)

        # 2. Compile and execute the pure JAX function
        if self._jitted_fn is None:
            self._jitted_fn = jax.jit(self.fn)

        return self._jitted_fn(*args, **kwargs)


class AxiomGradWrapper:
    def __init__(self, fn, has_aux=False, is_value_and_grad=False):
        self.fn = fn
        self.has_aux = has_aux
        self.is_value_and_grad = is_value_and_grad

    def __call__(self, *args, **kwargs):
        import jax.numpy as jnp
        import jax

        # --- 1. THE GHOST PASS BYPASS ---
        # If we are eagerly initializing, DO NOT let JAX trace!
        if getattr(compiler_state, 'is_initializing', False):
            out = self.fn(*args, **kwargs)

            # Create a mock PyTree of zero-gradients so `apply_updates`
            # doesn't crash during the rest of the eager step!
            m = args[0]  # The model being differentiated
            mock_params = {k: jnp.zeros_like(v.unwrap() if hasattr(v, 'unwrap') else v) for k, v in m.params.items()}
            mock_grads = AxiomModel(m.fn, mock_params)

            if self.is_value_and_grad:
                return out, mock_grads
            return mock_grads

        # --- 2. PURE XLA COMPILATION ---
        def jax_compatible_fn(*f_args, **f_kwargs):
            out = self.fn(*f_args, **f_kwargs)
            if self.has_aux:
                loss_tensor, aux = out
                return jnp.sum(loss_tensor.unwrap()), out
            else:
                loss_tensor = out
                return jnp.sum(loss_tensor.unwrap()), out

        if self.is_value_and_grad:
            (_, smuggled_out), grads = jax.value_and_grad(jax_compatible_fn, has_aux=True)(*args, **kwargs)
            return smuggled_out, grads
        else:
            grads, smuggled_out = jax.grad(jax_compatible_fn, has_aux=True)(*args, **kwargs)
            if self.has_aux:
                return grads, smuggled_out[1]
            return grads


# ==========================================
# 3. THE PUBLIC API
# ==========================================

def jit(fn):
    """Compiles an Axiom training step or mathematical function."""
    return AxiomJitWrapper(fn)


def value_and_grad(fn=None, has_aux=False):
    """Returns (loss, gradients). Operates transparently on AxiomModels."""
    if fn is None:
        return lambda f: AxiomGradWrapper(f, has_aux=has_aux, is_value_and_grad=True)
    return AxiomGradWrapper(fn, has_aux=has_aux, is_value_and_grad=True)


def grad(fn=None, has_aux=False):
    """Returns only gradients."""
    if fn is None:
        return lambda f: AxiomGradWrapper(f, has_aux=has_aux, is_value_and_grad=False)
    return AxiomGradWrapper(fn, has_aux=has_aux, is_value_and_grad=False)


def apply_updates(model: AxiomModel, grads: Any, optimizer: Any, opt_state: Any) -> Tuple[AxiomModel, Any]:
    """
    Orthogonal, functional optimizer step.
    Returns a newly updated AxiomModel PyTree and the new optimizer state.
    """
    import optax

    # QoL check: If the user surgically modified grads and passed a raw dictionary,
    # we dynamically wrap it back into an AxiomModel so Optax's tree_map doesn't crash!
    if isinstance(grads, dict):
        grads = AxiomModel(model.fn, grads)

    # 1. Initialize optimizer state dynamically if this is the first step
    # We pass the full PyTree model here!
    if opt_state is None:
        opt_state = optimizer.init(model)

    # 2. Calculate updates and apply them natively on the PyTrees!
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = optax.apply_updates(model, updates)

    # 3. Return the new Immutable PyTree
    return new_model, new_opt_state


def to_jax(model: AxiomModel):
    """
    Converts an AxiomModel into a pure JAX (params, apply_fn) paradigm.
    This allows for fine-grained manual control (like jax.vjp, custom gradients, etc.)
    """
    if not model.is_initialized:
        raise ValueError("AxiomModel must be initialized (run an eager forward pass) before converting to pure JAX.")

    # The parameters are already a flat dictionary of raw jax.Arrays!
    params = model.params.copy()

    def apply_fn(params_dict, *args, **kwargs):
        from .core import decay_monads, compiler_state

        args = decay_monads(args)
        kwargs = decay_monads(kwargs)

        # 1. Snapshot the state
        prev_params = getattr(compiler_state, 'params', {})
        prev_counter = getattr(compiler_state, 'param_counter', 0)
        prev_frames = compiler_state.active_frames.copy()
        prev_calls = compiler_state.func_calls.copy()

        # 2. Inject the pure functional parameters
        compiler_state.params = params_dict
        compiler_state.param_counter = 0
        compiler_state.active_frames.clear()
        compiler_state.func_calls.clear()

        # 3. Execute and safely restore
        try:
            res = model.fn(*args, **kwargs)
        finally:
            compiler_state.param_counter = prev_counter
            compiler_state.active_frames = prev_frames
            compiler_state.func_calls = prev_calls
            compiler_state.params = prev_params

        return res

    return params, apply_fn