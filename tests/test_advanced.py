import pytest
import jax
import jax.numpy as jnp
from axiom import ax, Tensor, Bundle, wrap, nn, init
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


# ==========================================
# 1. TEST: Dynamic Pointwise & Axis Chaining
# ==========================================
def test_pure_pointwise():
    print("--- Testing Pure Pointwise & Chaining ---")
    x = init.normal(ax.b(2), ax.d(4))

    # Pure tensor pointwise (no axis needed)
    y1 = x.silu()

    # Targeted pointwise
    y2 = x.d.silu()

    print(f"Base x: {x.topology}")
    print(f"x.silu(): {y1.topology}")
    print(f"x.d.silu(): {y2.topology}\n")
    assert x.topology == y1.topology == y2.topology


# ==========================================
# 2. TEST: Multi-Axis Contraction (ViT / 2D)
# ==========================================
def test_multi_axis_contraction():
    print("--- Testing Multi-Axis Contraction ---")
    # Simulate a 2D image patch layout
    scores = init.normal(ax.b(2), ax.h(8), ax.w(8))
    values = init.normal(ax.b(2), ax.h(8), ax.w(8), ax.d(16))

    # Contract across BOTH spatial axes simultaneously
    out = scores.h.w @ values

    print(f"Scores: {scores.topology}")
    print(f"Values: {values.topology}")
    print(f"scores.h.w @ values -> {out.topology}\n")
    assert out.topology == (ax.b(2), ax.d(16))


# ==========================================
# 3. TEST: Bundle Renaming & Processing
# ==========================================
def test_bundle_operations():
    print("--- Testing Bundle Rename & Math ---")
    x = init.normal(ax.b(2), ax.s(4), ax.d(8))

    # 1. Project into 3 distinct tensors
    q, k, v = (x & x & x).d.proj(bias=False)

    # 2. Bundle NN operations
    q, k = (q & k).d.rms_norm()

    # 3. Multi-Axis Bundle Rename!
    k, v = (k & v).s.rename(ax.sk, ax.sv)

    print(f"q: {q.topology}")
    print(f"k: {k.topology}")
    print(f"v: {v.topology}\n")

    assert k.topology == (ax.b(2), ax.sk(4), ax.d(8))
    assert v.topology == (ax.b(2), ax.sv(4), ax.d(8))


# ==========================================
# 4. TEST: Bundle Joining (Gated RNN block)
# ==========================================
def test_bundle_join():
    print("--- Testing Bundle Join ---")
    x = init.normal(ax.b(2), ax.d(16))
    h = init.normal(ax.b(2), ax.d(16))

    # Join x and h along the 'd' dimension!
    # Expected: d(16) + d(16) -> d(32)
    joined = (x & h).d.join()

    # Project the 32-dim vector back down to 16
    out = joined.d.proj(ax.d(16)).d.silu()

    print(f"x: {x.topology}, h: {h.topology}")
    print(f"(x & h).d.join() -> {joined.topology}")
    print(f"Projected back -> {out.topology}\n")

    assert joined.topology == (ax.b(2), ax.d(32))
    assert out.topology == (ax.b(2), ax.d(16))


def test_multiaxis_contraction():
    print("--- Testing Multi-Axis Contraction (3D Attention) ---")
    # b: batch, s: sequence, sk1: spatial 1, sk2: spatial 2, d: feature
    q = init.normal(ax.b(2), ax.s(4), ax.d(8))
    k1 = init.normal(ax.b(2), ax.sk1(4), ax.d(8))
    k2 = init.normal(ax.b(2), ax.sk2(4), ax.d(8))
    v = init.normal(ax.b(2), ax.sk1(4), ax.sk2(4), ax.d(8))

    # 1. Compute bilinear scores: (q @ k1) * (q @ k2)
    # q @ k1 -> [b, s, sk1]
    # q @ k2 -> [b, s, sk2]
    # Result -> [b, s, sk1, sk2]
    scores = (q @ k1) * (q @ k2)

    print(f"Scores Topology: {scores.topology}")
    assert scores.topology == (ax.b(2), ax.s(4), ax.sk1(4), ax.sk2(4))

    # 2. Multi-axis contraction: scores.sk1.sk2 @ v
    # This should contract sk1 and sk2 simultaneously!
    # [b, s, sk1, sk2] @ [b, sk1, sk2, d] -> [b, s, d]
    out = scores.sk1.sk2 @ v

    print(f"Output Topology: {out.topology}\n")
    assert out.topology == (ax.b(2), ax.s(4), ax.d(8))


def test_tensor_recursion():
    print("--- Testing Tensor.apply_n (Weight-Tied Depth) ---")
    x = init.normal(ax.b(2), ax.s(4), ax.d(16))

    def weight_tied_block(carry: Tensor) -> Tensor:
        # A simple projection tied across all iterations
        return (carry + carry.d.proj().d.silu()).d.rms_norm()

    # Apply the block 8 times recursively
    # In XLA, this is a single 'scan' loop
    final_x = x.apply_n(weight_tied_block, times=8)

    print(f"Input: {x.topology}")
    print(f"Output after 8 recursive steps: {final_x.topology}\n")

    assert final_x.topology == x.topology


def test_bundle_recursion():
    print("--- Testing Bundle.apply_n (SSM / RNN State) ---")
    x = init.normal(ax.b(2), ax.d(16))
    h = init.zeros(ax.b(2), ax.h(32))  # Hidden state

    def rnn_step(states: Bundle) -> Bundle:
        curr_x, curr_h = states.tensors

        # 1. Update hidden state: h = tanh(Wx + Uh)
        # We explicitly rename 'h' to 'd' to align their spatial dimensions!
        curr_h_aligned = curr_h.h.rename(ax.d(32))

        # Now the single-axis join cleanly stacks them along 'd'
        joined = (curr_x & curr_h_aligned).d.join()

        # Project the joined 48-dim 'd' vector back into a 32-dim 'h' vector
        next_h = joined.d.proj(ax.h(32)).h.tanh()

        # 2. Update x: x = x + proj(h)
        next_x = curr_x + next_h.h.proj(ax.d(16))

        return next_x & next_h

    # Carry both (x & h) through 10 iterations
    out_x, out_h = (x & h).apply_n(rnn_step, times=10)

    print(f"Final x: {out_x.topology}")
    print(f"Final h: {out_h.topology}\n")

    assert out_x.topology == (ax.b(2), ax.d(16))
    assert out_h.topology == (ax.b(2), ax.h(32))