import pytest
import jax
import jax.numpy as jnp
from axiom import ax, Tensor, Bundle, wrap, nn, init
from axiom.core import compiler_state


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


import jax.numpy as jnp
from axiom.core import Tensor, ax
from axiom.state import state
import axiom.init as ax_init


# ==========================================
# 8. TEST: Initializer DSL Math
# ==========================================
def test_initializer_math():
    print("--- Testing Initializer DSL Math ---")
    compiler_state.params.clear()  # Reset state

    # 1. Multiply an initializer by a scalar
    custom_init = ax_init.ones * 5.0

    # 2. Use it directly to generate a tensor
    t = custom_init(ax.d(4))

    print(f"Generated Tensor: {t.unwrap()}")
    assert jnp.allclose(t.unwrap(), 5.0)
    print("Initializer Math works flawlessly!\n")


# ==========================================
# 9. TEST: Implicit Learnable Parameters (.bias / .gate)
# ==========================================
def test_implicit_parameters():
    print("--- Testing .bias() and .gate() with Invisible Scope ---")
    compiler_state.params.clear()

    # FIX: Use the new integer counter instead of the old dictionary!
    compiler_state.param_counter = 0

    x = Tensor(jnp.ones((2, 16)), ax.b(2), ax.d(16))

    def gated_residual_block(t):
        t = t.d.gate(init=ax_init.ones * 0.5)
        t = t.d.bias(init=ax_init.ones * 2.0)
        return t

    out = gated_residual_block(x)

    print(f"Output shape: {out.topology}")
    print(f"Output values (expected 2.5): {out.unwrap()[0, 0]}")

    assert jnp.allclose(out.unwrap(), 2.5)

    # FIX: Read from compiler_state.params instead of state.params!
    print(f"State Manager Params: {list(compiler_state.params.keys())}")
    assert "gated_residual_block/gate_0" in compiler_state.params
    assert "gated_residual_block/bias_1" in compiler_state.params
    print("Implicit parameters and Invisible Scope passed!\n")


# ==========================================
# 10. TEST: Bundle Parallel Tying
# ==========================================
def test_bundle_tied_gates():
    print("--- Testing Bundle .gate() and .bias() with Global Tying ---")
    compiler_state.params.clear()

    x = Tensor(jnp.ones((2, 16)), ax.b(2), ax.d(16))
    h = Tensor(jnp.ones((2, 16)), ax.b(2), ax.d(16))

    # Apply a globally tied gate and bias to BOTH tensors simultaneously
    out_x, out_h = (x & h).d.gate(tie="@shared_gate", init=ax_init.ones * 2.0) \
        .d.bias(tie="@shared_bias", init=ax_init.ones * 3.0)

    assert jnp.allclose(out_x.unwrap(), 5.0)
    assert jnp.allclose(out_h.unwrap(), 5.0)

    # FIX: Read from compiler_state.params instead of state.params!
    print(f"State Manager Params: {list(compiler_state.params.keys())}")
    assert "shared_gate" in compiler_state.params
    assert "shared_bias" in compiler_state.params
    assert len(compiler_state.params) == 2
    print("Bundle parallel gating passed!\n")


def test_scalar_indexing():
    print("--- Testing Scalar Indexing ---")
    # b=2, d=10
    x = init.normal(ax.b(2), ax.d(10))

    first = x.d[0]
    last = x.d[-1]
    middle = x.d[x.d // 2]

    print(f"Original: {x.topology}")
    print(f"First  x.d[0]       -> {first.topology}")
    print(f"Last   x.d[-1]      -> {last.topology}")
    print(f"Middle x.d[x.d//2]  -> {middle.topology}")

    # Assert the 'd' axis was physically removed
    assert first.topology == (ax.b(2),)
    assert last.topology == (ax.b(2),)
    assert middle.topology == (ax.b(2),)
    print("Scalar indexing passed!\n")

from axiom.core import SlicedMonad
def test_sliced_monad_scalar_math_preserves_patch():
    ax.d = ax("d", 4)
    x = wrap(jnp.array([1.0, 2.0, 3.0, 4.0]), ax.d)

    out = (x.d[:2] * 10.0)[:]

    assert isinstance(out, Tensor)
    assert out.topology == (ax.d,)
    assert out.topology[0].size == 4
    assert jnp.allclose(out.unwrap(), jnp.array([10.0, 20.0, 3.0, 4.0]))

def test_sliced_monad_plain_tensor_math_preserves_patch():
    ax.d = ax("d", 4)
    x = wrap(jnp.array([1.0, 2.0, 3.0, 4.0]), ax.d)
    delta = wrap(jnp.array([10.0, 20.0]), ax.d(2))

    out = (x.d[:2] + delta)[:]

    assert isinstance(out, Tensor)
    assert out.topology == (ax.d,)
    assert jnp.allclose(out.unwrap(), jnp.array([11.0, 22.0, 3.0, 4.0]))

def test_sliced_monad_cross_chunk_math_dissolves():
    ax.d = ax("d", 4)
    x = wrap(jnp.array([1.0, 2.0, 3.0, 4.0]), ax.d)

    left = x.d[:2]
    right = x.d[2:]

    result = left * right

    assert isinstance(left, SlicedMonad)
    assert isinstance(right, SlicedMonad)
    assert isinstance(result, Tensor)
    assert result.topology[0].size == 2
    assert jnp.allclose(result.unwrap(), jnp.array([3.0, 8.0]))

def test_sliced_monad_cross_chunk_math_has_no_patch_commit():
    ax.d = ax("d", 4)
    x = wrap(jnp.array([1.0, 2.0, 3.0, 4.0]), ax.d)

    result = x.d[:2] * x.d[2:]

    assert isinstance(result, Tensor)

    with pytest.raises(Exception):
        _ = result[:]

def test_sliced_monad_default_proj_is_patch_safe():
    ax.d = ax("d", 4)
    x = wrap(jnp.ones(4), ax.d)

    out = x.d[:2].proj()[:]

    assert isinstance(out, Tensor)
    assert out.topology == (ax.d,)
    assert out.topology[0].size == 4

def test_sliced_monad_explicit_proj_new_axis_is_not_patch_safe():
    ax.d = ax("d", 4)
    ax.d2 = ax("d2", 2)
    x = wrap(jnp.ones(4), ax.d)

    y = x.d[:2].proj(ax.d2)

    assert isinstance(y, SlicedMonad)

    with pytest.raises(ValueError, match="Explicit proj|unsafe|topology"):
        _ = y[:]

def test_sliced_monad_explicit_proj_same_axis_is_not_patch_safe():
    ax.d = ax("d", 4)
    x = wrap(jnp.ones(4), ax.d)

    y = x.d[:2].proj(ax.d(2))

    assert isinstance(y, SlicedMonad)

    with pytest.raises(ValueError, match="Explicit proj|unsafe|topology"):
        _ = y[:]

def test_sliced_monad_explicit_proj_same_name_wrong_size_is_not_patch_safe():
    ax.d = ax("d", 4)
    x = wrap(jnp.ones(4), ax.d)

    y = x.d[:2].proj(ax.d(128))

    assert isinstance(y, SlicedMonad)

    with pytest.raises(ValueError, match="Explicit proj|unsafe|topology"):
        _ = y[:]

def test_sliced_monad_rename_is_not_patch_safe():
    ax.d = ax("d", 4)
    ax.h = ax("h", 2)
    x = wrap(jnp.ones(4), ax.d)

    y = x.d[:2].d.rename(ax.h)

    assert isinstance(y, SlicedMonad)

    with pytest.raises(ValueError, match="rename|unsafe|topology"):
        _ = y[:]

def test_sliced_monad_pad_is_not_patch_safe():
    ax.d = ax("d", 4)
    x = wrap(jnp.ones(4), ax.d)

    y = x.d[:2].d.pad((1, 0))

    assert isinstance(y, SlicedMonad)

    with pytest.raises(ValueError, match="pad|unsafe|topology"):
        _ = y[:]

def test_sliced_monad_reduction_decays_to_tensor():
    ax.d = ax("d", 4)
    x = wrap(jnp.array([1.0, 2.0, 3.0, 4.0]), ax.d)

    y = x.d[:2].d.sum()

    assert isinstance(y, Tensor)
    assert y.topology == ()
    assert jnp.allclose(y.unwrap(), 3.0)

def test_sliced_monad_axis_chaining_preserves_patch_context():
    ax.s = ax("s", 4)
    ax.d = ax("d", 3)

    x = wrap(jnp.ones((4, 3)), ax.s, ax.d)

    y = x.s[:2].d.bias()

    assert isinstance(y, SlicedMonad)

    out = y[:]

    assert isinstance(out, Tensor)
    assert out.topology == (ax.s, ax.d)
    assert out.unwrap().shape == (4, 3)

def test_sliced_monad_nested_slice_is_not_commit_compatible():
    ax.s = ax("s", 4)
    ax.d = ax("d", 4)

    x = wrap(jnp.ones((4, 4)), ax.s, ax.d)

    y = x.s[:2].d[:2]

    assert isinstance(y, SlicedMonad)

    with pytest.raises(ValueError, match="topology"):
        _ = y[:]

def test_gather_accepts_sliced_monad_index():
    ax.s = ax("s", 4)
    ax.vocab = ax("vocab", 5)
    ax.d = ax("d", 1)

    tokens = wrap(jnp.array([1, 0, 3, 2]), ax.s)

    embeddings = wrap(
        jnp.array([[10.0], [20.0], [30.0], [40.0], [50.0]]),
        ax.vocab,
        ax.d,
    )

    gathered = embeddings.vocab[tokens.s[:2]].gather()

    assert isinstance(gathered, Tensor)
    assert gathered.topology == (ax.s(2), ax.d)
    assert jnp.allclose(gathered.unwrap(), jnp.array([[20.0], [10.0]]))

def test_decay_monads_turns_sliced_monad_into_tensor():
    from axiom.core import decay_monads

    ax.d = ax("d", 4)
    x = wrap(jnp.array([1.0, 2.0, 3.0, 4.0]), ax.d)

    lazy = x.d[:2]
    decayed = decay_monads(lazy)

    assert isinstance(lazy, SlicedMonad)
    assert isinstance(decayed, Tensor)
    assert decayed.topology == (ax.d(2),)
    assert jnp.allclose(decayed.unwrap(), jnp.array([1.0, 2.0]))

def test_decay_monads_recurses_through_containers():
    from axiom.core import decay_monads

    ax.d = ax("d", 4)
    x = wrap(jnp.array([1.0, 2.0, 3.0, 4.0]), ax.d)

    nested = {
        "a": x.d[:2],
        "b": [x.d[2:], 123],
    }

    out = decay_monads(nested)

    assert isinstance(out["a"], Tensor)
    assert isinstance(out["b"][0], Tensor)
    assert out["b"][1] == 123
    assert jnp.allclose(out["a"].unwrap(), jnp.array([1.0, 2.0]))
    assert jnp.allclose(out["b"][0].unwrap(), jnp.array([3.0, 4.0]))

def test_sliced_monad_step_slice_commit_is_rejected():
    ax.d = ax("d", 6)
    x = wrap(jnp.arange(6.0), ax.d)

    y = x.d[::2] * 10.0

    assert isinstance(y, SlicedMonad)

    with pytest.raises(ValueError, match="step|contiguous"):
        _ = y[:]

def test_sliced_monad_negative_slice_commit_is_allowed():
    ax.d = ax("d", 6)
    x = wrap(jnp.arange(6.0), ax.d)

    y = (x.d[-2:] * 10.0)[:]

    assert isinstance(y, Tensor)
    assert y.topology == (ax.d,)
    assert jnp.allclose(y.unwrap(), jnp.array([0.0, 1.0, 2.0, 3.0, 40.0, 50.0]))

def test_sliced_monad_negative_start_stop_commit_is_allowed():
    ax.d = ax("d", 6)
    x = wrap(jnp.arange(6.0), ax.d)

    y = (x.d[-4:-1] + 100.0)[:]

    assert isinstance(y, Tensor)
    assert y.topology == (ax.d,)
    assert jnp.allclose(y.unwrap(), jnp.array([0.0, 1.0, 102.0, 103.0, 104.0, 5.0]))

def test_sliced_monad_step_slice_commit_is_rejected():
    ax.d = ax("d", 6)
    x = wrap(jnp.arange(6.0), ax.d)

    y = x.d[::2] * 10.0

    assert isinstance(y, SlicedMonad)

    with pytest.raises(ValueError, match="step|contiguous"):
        _ = y[:]