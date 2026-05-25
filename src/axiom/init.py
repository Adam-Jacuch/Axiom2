import jax
import jax.numpy as jnp
from .core import wrap, Axis, Tensor  # Import Axis for strict routing

def _resolve_axes(axes, like):
    """Helper to extract topology from a 'like' tensor or targeted tensor."""
    if like is not None:
        if hasattr(like, 'target_axes'):
            return like.target_axes  # User passed like=x.d
        elif hasattr(like, 'topology'):
            return like.topology     # User passed like=x
        else:
            raise ValueError("The 'like' argument must be a Tensor or TargetedTensor.")
    return axes

def _generate(init_obj, *axes, like=None):
    from .state import state
    resolved_axes = _resolve_axes(axes, like)
    shape = [a.size for a in resolved_axes]
    # Pass the State Manager's key and the shape directly into the apply_fn
    raw_array = init_obj.apply_fn(state.next_key(), shape)
    return wrap(raw_array, *resolved_axes)

class Initializer:
    """A mathematically overloadable and context-aware initializer."""
    def __init__(self, apply_fn):
        self.apply_fn = apply_fn

    def __call__(self, *args, **kwargs):
        # --- STRICT ROUTING ---
        is_tensor_gen = False
        if 'like' in kwargs:
            is_tensor_gen = True
        elif len(args) == 0:
            is_tensor_gen = True  # Creating a scalar tensor
        elif len(args) > 0 and isinstance(args[0], Axis):
            is_tensor_gen = True  # Strict check! Prevents JAX arrays from triggering this.

        if is_tensor_gen:
            return _generate(self, *args, **kwargs)

        # 2. Otherwise, it's the State Manager requesting raw JAX arrays!
        key, shape = args[0], args[1]
        return self.apply_fn(key, shape, **kwargs)

    def __mul__(self, scalar):
        return Initializer(lambda k, s, **kw: self.apply_fn(k, s, **kw) * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, scalar):
        return Initializer(lambda k, s, **kw: self.apply_fn(k, s, **kw) + scalar)

    def __radd__(self, scalar):
        return self.__add__(scalar)

# --- Built-in Initializers ---
zeros = Initializer(lambda k, s, **kw: jnp.zeros(s))
ones = Initializer(lambda k, s, **kw: jnp.ones(s))
normal = Initializer(lambda k, s, **kw: jax.random.normal(k, s))
uniform = Initializer(lambda k, s, **kw: jax.random.uniform(k, s))
xavier = Initializer(lambda k, s, fan_in=1, fan_out=1, **kw: jax.random.normal(k, s) * jnp.sqrt(2.0 / (fan_in + fan_out)))
he = Initializer(lambda k, s, fan_in=1, **kw: jax.random.normal(k, s) * jnp.sqrt(2.0 / fan_in))


def arange(*args) -> Tensor:
    """
    Axiom wrapper for jnp.arange.
    The final argument MUST be the target Axis.

    Usage:
        init.arange(10, ax.s)
        init.arange(0, 10, 2, ax.h_dim)
    """
    if not args or not isinstance(args[-1], Axis):
        raise ValueError("The last argument to init.arange must be an Axiom Axis.")

    target_ax = args[-1]
    jnp_args = args[:-1]

    # Generate the raw JAX array
    raw_array = jnp.arange(*jnp_args)

    # Safety Check: Did the generated array match the expected axis size?
    if target_ax.size is not None and target_ax.size != raw_array.size:
        raise ValueError(
            f"init.arange produced {raw_array.size} elements, but "
            f"axis '{target_ax.name}' expects {target_ax.size} elements."
        )

    # If the user passed an unsized axis, infer it dynamically!
    final_ax = target_ax(raw_array.size) if target_ax.size is None else target_ax

    return Tensor(raw_array, final_ax)