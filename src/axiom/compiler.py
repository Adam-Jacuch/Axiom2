import jax
from functools import wraps
from .core import compiler_state, Tensor, decay_monads

from typing import Callable, Any, TypeVar, Union

# Create a generic type variable for functions
F = TypeVar('F', bound=Callable[..., Any])

# Ensure the tracer routing attribute exists
compiler_state.step_params = None


class AxiomFunction:
    """The AOT Compiled Execution Graph."""

    def __init__(self, fn):
        self.fn = fn
        self.params = {}
        self.is_initialized = False
        self._jitted_fn = None

    def __call__(self, *args, **kwargs):
        # Lazy slices used as ordinary values should decay before tracing/init.
        args = decay_monads(args)
        kwargs = decay_monads(kwargs)

        if not self.is_initialized:
            compiler_state.is_initializing = True
            compiler_state.params = {}
            compiler_state.param_counter = 0

            self.fn(*args, **kwargs)

            self.params = compiler_state.params.copy()
            self.is_initialized = True
            compiler_state.is_initializing = False

            def pure_fn(params, *p_args, **p_kwargs):
                p_args = decay_monads(p_args)
                p_kwargs = decay_monads(p_kwargs)

                compiler_state.params = params
                compiler_state.param_counter = 0
                return self.fn(*p_args, **p_kwargs)

            self._jitted_fn = jax.jit(pure_fn)

        # Route JAX gradient tracers into the fast path if we are in a training step.
        active_params = compiler_state.step_params if compiler_state.step_params is not None else self.params

        return self._jitted_fn(active_params, *args, **kwargs)

    def get_state(self) -> dict:
        if not self.is_initialized:
            raise RuntimeError("AxiomFunction must be called once to initialize.")
        return self.params

    def vjp(self, loss: Tensor):
        """Syntactic sugar."""
        pass

    def step(self):
        """Syntactic sugar."""
        pass


def axiom_jit(fn):
    return AxiomFunction(fn)


def axiom_step(
    model: Union[Callable[..., Any], Any],
    optimizer: Any
) -> Callable[[F], F]:
    """Compiles a training step into a pure JAX execution graph."""
    import optax

    def decorator(step_fn):
        opt_state = None
        _jitted_train_step = None

        @wraps(step_fn)
        def wrapper(*args, **kwargs):
            nonlocal opt_state, _jitted_train_step

            args = decay_monads(args)
            kwargs = decay_monads(kwargs)

            if not model.is_initialized:
                step_fn(*args, **kwargs)

            if opt_state is None:
                opt_state = optimizer.init(model.params)

                def pure_train_step(params, opt_state_inner, *p_args):
                    p_args = decay_monads(p_args)

                    import jax.numpy as jnp

                    def loss_fn(p):
                        # INJECT TRACERS: Force the model to use the gradient tape.
                        compiler_state.step_params = p

                        loss_tensor = step_fn(*p_args)

                        # CLEAN UP
                        compiler_state.step_params = None

                        return jnp.sum(loss_tensor.unwrap())

                    loss_val, grads = jax.value_and_grad(loss_fn)(params)

                    updates, new_opt_state = optimizer.update(grads, opt_state_inner, params)
                    new_params = optax.apply_updates(params, updates)

                    return loss_val, new_params, new_opt_state

                _jitted_train_step = jax.jit(pure_train_step)

            loss_val, new_params, opt_state = _jitted_train_step(model.params, opt_state, *args)

            model.params = new_params

            return loss_val

        return wrapper

    return decorator