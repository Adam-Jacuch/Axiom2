import jax
import jax.numpy as jnp
from .core import wrap

# The hidden global state
_GLOBAL_KEY = jax.random.PRNGKey(42)

def _next_key():
    """Splits the key and updates the global state silently."""
    global _GLOBAL_KEY
    _GLOBAL_KEY, subkey = jax.random.split(_GLOBAL_KEY)
    return subkey

def zeros(*axes):
    return wrap(jnp.zeros([a.size for a in axes]), *axes)

def ones(*axes):
    return wrap(jnp.ones([a.size for a in axes]), *axes)

def normal(*axes):
    # Now it grabs a freshly split key every single time!
    shape = [a.size for a in axes]
    return wrap(jax.random.normal(_next_key(), shape), *axes)