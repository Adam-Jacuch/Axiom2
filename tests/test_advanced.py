import pytest
import jax
import jax.numpy as jnp
from axiom import ax, Tensor, wrap, nn, init
from axiom.core import SlicedMonad


def test_transparent_math_proxies():
    """Tests that operations on TargetedTensors forward to the underlying Tensor."""
    ax.d = ax("d", 128)
    x = wrap(jnp.ones(128), ax.d)

    # 1. TargetedTensor * scalar
    res_mul = x.d * 2.0
    assert isinstance(res_mul, Tensor)
    assert jnp.allclose(res_mul.unwrap(), 2.0)

    # 2. TargetedTensor + TargetedTensor
    res_add = x.d + x.d
    assert isinstance(res_add, Tensor)
    assert jnp.allclose(res_add.unwrap(), 2.0)


def test_sliced_monad_stitching():
    """Tests that [:] correctly stitches a modified chunk back into the original topology."""
    ax.d = ax("d", 4)
    # [1.0, 1.0, 1.0, 1.0]
    x = wrap(jnp.ones(4), ax.d)

    # Slice the first half, multiply by 5, and stitch it back!
    stitched = (x.d[:2] * 5.0)[:]

    assert isinstance(stitched, Tensor)
    assert stitched.topology[0].size == 4

    # Should be [5.0, 5.0, 1.0, 1.0]
    expected = jnp.array([5.0, 5.0, 1.0, 1.0])
    assert jnp.allclose(stitched.unwrap(), expected)


def test_sliced_monad_cross_chunk_math():
    """Tests the SwiGLU magic: multiplying two chunks drops the monad and returns a pure Tensor."""
    ax.d = ax("d", 4)
    # [1.0, 2.0, 3.0, 4.0]
    x = wrap(jnp.array([1.0, 2.0, 3.0, 4.0]), ax.d)

    left = x.d[:2]  # [1.0, 2.0]
    right = x.d[2:]  # [3.0, 4.0]

    assert isinstance(left, SlicedMonad)

    # Multiply cross-chunks!
    # [1.0 * 3.0, 2.0 * 4.0] = [3.0, 8.0]
    result = left * right

    # The monad should dissolve, returning a pure Tensor of size 2
    assert isinstance(result, Tensor)
    assert result.topology[0].size == 2
    assert jnp.allclose(result.unwrap(), jnp.array([3.0, 8.0]))


def test_topological_pad():
    """Tests that padding safely increases the axis size and places values correctly."""
    ax.s = ax("s", 3)
    x = wrap(jnp.ones(3), ax.s)

    # Pad 1 on the left, 2 on the right
    padded = x.s.pad((1, 2), fill=0.0)

    assert isinstance(padded, Tensor)
    assert padded.topology[0].size == 6  # 1 + 3 + 2 = 6

    expected = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(padded.unwrap(), expected)


def test_fluent_context_gather():
    """Tests that Gather immediately resolves to a pure Tensor."""
    ax.s = ax("s", 2)
    ax.vocab = ax("vocab", 3)
    ax.d = ax("d", 1)

    # Tokens: [1, 0]
    tokens = wrap(jnp.array([1, 0]), ax.s)

    # Embeddings: [[9.0], [8.0], [7.0]]
    embed_raw = jnp.array([[9.0], [8.0], [7.0]])
    embeddings = wrap(embed_raw, ax.vocab, ax.d)

    # Target vocab, pass tokens, and gather!
    gathered = embeddings.vocab[tokens].gather()

    # Should be pure tensor! No [:] required.
    assert isinstance(gathered, Tensor)

    # Topology should be [s, d], vocab is destroyed
    assert gathered.topology == (ax.s, ax.d)

    # Expected output: [[8.0], [9.0]]
    expected = jnp.array([[8.0], [9.0]])
    assert jnp.allclose(gathered.unwrap(), expected)


def test_rms_norm_programmatic_retargeting():
    """Tests that our fix to rms_norm successfully retargets the squared pure tensor."""
    ax.d = ax("d", 128)
    x = wrap(jnp.ones(128) * 2.0, ax.d)

    # If the targeted.pw retargeting works, this will not crash!
    normalized = nn.rms_norm(x.d)

    assert isinstance(normalized, Tensor)
    assert normalized.topology[0].size == 128


def test_true_random_init():
    """Tests that our hidden state tracker yields different arrays sequentially."""
    ax.d = ax("d", 100)

    # Grab two sequential normal distributions
    a = init.normal(ax.d)
    b = init.normal(ax.d)

    # Because of our global PRNG key split, they should NOT be equal!
    assert not jnp.allclose(a.unwrap(), b.unwrap())