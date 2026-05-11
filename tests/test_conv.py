import pytest
import jax.numpy as jnp
from axiom import ax, Tensor, wrap, axiom_jit


def test_linear_projection_with_bias():
    """Tests that biases are correctly initialized and broadcasted."""

    @axiom_jit
    def linear_layer(x: Tensor):
        # Project from 16 to 32, with a bias
        return x.in_d.proj(ax.out_d(32), bias=True)

    # Input: (batch=8, in_d=16)
    x = wrap(jnp.ones((8, 16)), ax.b(8), ax.in_d(16))
    out = linear_layer(x)

    # 1. Check the topology
    assert out.topology == (ax.b(8), ax.out_d(32))
    assert out.unwrap().shape == (8, 32)

    # 2. Verify the parameters (1 for Weights, 1 for Bias)
    state = linear_layer.get_state()
    assert len(state) == 2
    assert "param_0" in state  # Weights: (16, 32)
    assert "param_1" in state  # Bias: (32,) -> broadcasted automatically!


def test_sequence_unfold():
    """Tests the sliding window creation for sequences."""
    # Input: (batch=2, seq=10)
    x = wrap(jnp.arange(20).reshape(2, 10), ax.b(2), ax.seq(10))

    # Unfold with a kernel of 3, step of 1
    windows = x.seq.unfold(ax.kernel(3))

    # Expected new sequence length: 10 - 3 + 1 = 8
    expected_seq_ax = ax("seq", 8)

    # 1. Check the new topology
    assert windows.topology == (ax.b(2), expected_seq_ax, ax.kernel(3))
    assert windows.unwrap().shape == (2, 8, 3)


def test_native_1d_convolution():
    """Combines unfold and proj to perform a 1D Convolution over a sequence."""

    @axiom_jit
    def conv1d_layer(x: Tensor):
        # 1. Unfold the sequence
        windows = x.seq.unfold(ax.kernel(3))

        # 2. Project over the channel and kernel axes simultaneously!
        return windows.in_c.kernel.proj(ax.out_c(64), bias=True)

    # Input: (batch=4, seq=32, in_c=16)
    x = wrap(jnp.ones((4, 32, 16)), ax.b(4), ax.seq(32), ax.in_c(16))
    out = conv1d_layer(x)

    # Expected new sequence length: 32 - 3 + 1 = 30
    expected_seq_ax = ax("seq", 30)

    # Check the final topology! It should be strictly (batch, seq, out_c)
    assert out.topology == (ax.b(4), expected_seq_ax, ax.out_c(64))
    assert out.unwrap().shape == (4, 30, 64)


if __name__ == "__main__":
    pytest.main(["-v", __file__])