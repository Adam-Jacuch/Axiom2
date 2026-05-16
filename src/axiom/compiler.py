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

        if compiler_state.is_initializing:
            # --- THE GHOST PASS ---
            # We are executing eagerly to trace shapes and allocate parameters.
            compiler_state.params = self.params

            res = self.fn(*args, **kwargs)

            self.params = compiler_state.params.copy()
            self.is_initialized = True
            return res
        else:
            # --- PURE XLA EXECUTION ---
            # Inject the active PyTree params into the framework's tracer context.
            # (We save the previous state so models can call other models safely!)
            prev_params = compiler_state.step_params
            compiler_state.step_params = self.params

            res = self.fn(*args, **kwargs)

            compiler_state.step_params = prev_params
            return res

    # --- NATIVE MODEL CALCULUS (For Meta-Learning) ---
    def __sub__(self, other):
        if isinstance(other, dict):  # Subtracting gradients
            new_params = jax.tree_util.tree_map(lambda p, g: p - g, self.params, other)
            return AxiomModel(self.fn, new_params)
        raise TypeError("Can only subtract parameter dictionaries (gradients) from an AxiomModel.")

    def __add__(self, other):
        if isinstance(other, dict):
            new_params = jax.tree_util.tree_map(lambda p, g: p + g, self.params, other)
            return AxiomModel(self.fn, new_params)
        raise TypeError("Can only add parameter dictionaries to an AxiomModel.")


# Tell JAX how to flatten and unflatten our model!
def _unflatten_model(aux, children):
    fn, is_init = aux  # Unpack the static auxiliary data
    params = children[0]  # Unpack the dynamic PyTree leaves (the dict)

    # Rebuild the model
    model = AxiomModel(fn, params)
    model.is_initialized = is_init  # Restore its true initialization state!
    return model


register_pytree_node(
    AxiomModel,
    lambda m: ((m.params,), (m.fn, m.is_initialized)),  # Flatten
    _unflatten_model  # Unflatten
)


# ==========================================
# 2. GHOST INITIALIZATION WRAPPERS
# ==========================================
def _trigger_ghost_pass(fn, *args, **kwargs):
    """Eagerly runs the function to auto-initialize any uninitialized AxiomModels."""
    # Find all AxiomModels in the inputs (searching flat args for simplicity)
    models = [arg for arg in jax.tree_util.tree_leaves(args) if isinstance(arg, AxiomModel)]

    if any(not m.is_initialized for m in models):
        # Run exactly once in initialization mode
        compiler_state.is_initializing = True
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

        # 1. Ghost Pass
        _trigger_ghost_pass(self.fn, *args, **kwargs)

        # 2. The Smuggling Function
        def jax_compatible_fn(*f_args, **f_kwargs):
            out = self.fn(*f_args, **f_kwargs)

            if self.has_aux:
                loss_tensor, aux = out
                # Give JAX the raw scalar, smuggle the original 'out' tuple via aux
                return jnp.sum(loss_tensor.unwrap()), out
            else:
                loss_tensor = out
                # Give JAX the raw scalar, smuggle the original Tensor via aux
                return jnp.sum(loss_tensor.unwrap()), out

        # 3. Execute JAX Gradient function (Forcing has_aux=True internally)
        if self.is_value_and_grad:
            # JAX returns ((raw_scalar, smuggled_out), grads)
            (_, smuggled_out), grads = jax.value_and_grad(jax_compatible_fn, has_aux=True)(*args, **kwargs)
            return smuggled_out, grads
        else:
            # JAX returns (grads, smuggled_out)
            grads, smuggled_out = jax.grad(jax_compatible_fn, has_aux=True)(*args, **kwargs)
            if self.has_aux:
                _, aux = smuggled_out
                return grads, aux
            else:
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