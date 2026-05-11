import pytest
import jax
import jax.numpy as jnp
from axiom import ax, Tensor, wrap, axiom_jit, nn


def test_layer_norm_and_state():
    ax.b = ax("b", 8)
    ax.s = ax("s", 128)
    ax.d = ax("d", 64)

    @axiom_jit
    def simple_block(x: Tensor):
        # Target the dimension axis for layer normalization!
        return x.d.pw(nn.layer_norm)

    dummy_x = wrap(jax.random.normal(jax.random.PRNGKey(0), (8, 128, 64)), ax.b, ax.s, ax.d)

    # Run the AOT trace
    out = simple_block(dummy_x)

    # 1. Check Topology
    assert out.topology == (ax.b, ax.s, ax.d)

    # 2. Check State Interception
    state = simple_block.get_state()

    # The compiler should have seamlessly caught the 'gamma' and 'beta' from nn.py!
    assert len(state) == 2
    assert "gamma" in state
    assert "beta" in state

    # Gamma and Beta should have the shape of the targeted axis (d=64)
    assert state["gamma"].shape == (64,)
    assert state["beta"].shape == (64,)


def test_jax_native_softmax_injection():
    ax.b = ax("b", 8)
    ax.vocab = ax("vocab", 1000)

    logits = wrap(jnp.ones((8, 1000)), ax.b, ax.vocab)

    # This proves `.pw()` successfully reads the `axis` arg from jax.nn.softmax
    # and automatically passes the index of `vocab`!
    probs = logits.vocab.pw(nn.softmax)

    assert probs.topology == (ax.b, ax.vocab)
    # If softmax worked over vocab, they should all sum to 1.0
    assert jnp.allclose(probs.vocab.sum().unwrap(), jnp.ones((8,)))