import jax
import jax.numpy as jnp
from .core import wrap

# The hidden global state!
_GLOBAL_KEY = jax.random.PRNGKey(42)

def _next_key():
    """Splits the key and updates the global state silently (Safely handles JIT!)."""
    global _GLOBAL_KEY
    new_key, subkey = jax.random.split(_GLOBAL_KEY)

    # Tracer Leak Prevention
    # If JAX is currently tracing the graph, the new_key will be a Tracer.
    # We MUST NOT assign a Tracer to a global Python variable, or JAX will crash.
    if not isinstance(new_key, jax.core.Tracer):
        _GLOBAL_KEY = new_key

    return subkey

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

def zeros(*axes, like=None):
    resolved_axes = _resolve_axes(axes, like)
    return wrap(jnp.zeros([a.size for a in resolved_axes]), *resolved_axes)

def ones(*axes, like=None):
    resolved_axes = _resolve_axes(axes, like)
    return wrap(jnp.ones([a.size for a in resolved_axes]), *resolved_axes)

def normal(*axes, like=None):
    resolved_axes = _resolve_axes(axes, like)
    shape = [a.size for a in resolved_axes]
    return wrap(jax.random.normal(_next_key(), shape), *resolved_axes)