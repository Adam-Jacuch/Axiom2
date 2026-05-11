import pytest
import jax.numpy as jnp
from axiom import ax, Tensor, wrap, axiom_jit


def test_axiom_jit_and_state_extraction():
    ax.b = ax("b", 16)
    ax.d = ax("d", 64)
    ax.d2 = ax("d2", 128)

    # Define a simple multi-layer block
    @axiom_jit
    def my_network(x: Tensor):
        # Implicitly initializes a (64, 128) weight matrix
        h = x.d.proj(ax.d2)
        # Implicitly initializes a second (128, 64) weight matrix
        return h.d2.proj(ax.d)

    # Wrap some dummy input
    dummy_x = wrap(jnp.ones((16, 64)), ax.b, ax.d)

    # 1. First call triggers the AOT Trace and JAX compilation
    output = my_network(dummy_x)

    assert output.topology == (ax.b, ax.d)
    assert output.unwrap().shape == (16, 64)

    # 2. Extract the PyTree state!
    state = my_network.get_state()

    # The compiler should have intercepted exactly two parameters
    assert len(state) == 2
    assert "param_0" in state
    assert "param_1" in state

    # Verify the shapes of the silently allocated weights
    assert state["param_0"].shape == (64, 128)
    assert state["param_1"].shape == (128, 64)


def test_weight_tying_in_compiler():
    ax.b = ax("b", 8)
    ax.d = ax("d", 32)
    ax.latent = ax("latent", 16)

    @axiom_jit
    def siamese_network(image_a: Tensor, image_b: Tensor):
        # Tie the weights so both projections share the exact same matrix!
        emb_a = image_a.d.proj(ax.latent, tie="shared_encoder")
        emb_b = image_b.d.proj(ax.latent, tie="shared_encoder")
        return emb_a, emb_b

    img_a = wrap(jnp.ones((8, 32)), ax.b, ax.d)
    img_b = wrap(jnp.ones((8, 32)), ax.b, ax.d)

    out_a, out_b = siamese_network(img_a, img_b)

    state = siamese_network.get_state()

    # The compiler should have only allocated ONE weight matrix because of the tie!
    assert len(state) == 1
    assert "shared_encoder" in state
    assert state["shared_encoder"].shape == (32, 16)