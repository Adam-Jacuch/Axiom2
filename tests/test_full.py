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
    final_x = x.repeat(weight_tied_block, times=8)

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
    out_x, out_h = (x & h).repeat(rnn_step, times=10)

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
    assert "gated_residual_block_0/gate_0" in compiler_state.params
    assert "gated_residual_block_0/bias_1" in compiler_state.params
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


# ==========================================
# 11. TEST: The Universal Dispatcher (Dynamic JAX Routing)
# ==========================================
def test_dynamic_jax_reduction():
    print("--- Testing Dynamic JAX Reduction (var / std) ---")
    # 'var' and 'std' are NOT explicitly defined in core.py.
    # They must be dynamically caught by TargetedTensor.__getattr__ and injected with the axis!
    x = init.normal(ax.b(2), ax.s(4), ax.d(8))

    # Target 's' and calculate variance
    out_var = x.s.var()

    assert isinstance(out_var, Tensor)
    # The 's' axis should be collapsed, leaving 'b' and 'd'
    assert out_var.topology == (ax.b(2), ax.d(8))
    print("Dynamic JAX reduction passed!\n")


def test_dynamic_pure_pointwise_base_tensor():
    print("--- Testing Pure Pointwise on Base Tensor ---")
    # Base Tensors should catch pure pointwise math without needing to target an axis
    x = wrap(jnp.array([-1.0, 0.0, 1.0]), ax.d(3))

    out_exp = x.exp()
    out_abs = x.abs()

    assert isinstance(out_exp, Tensor)
    assert out_exp.topology == (ax.d(3),)
    assert jnp.allclose(out_abs.unwrap(), jnp.array([1.0, 0.0, 1.0]))
    print("Base tensor dynamic pointwise passed!\n")


def test_mathematical_ambiguity_guard():
    print("--- Testing Mathematical Ambiguity Guard ---")
    x = init.normal(ax.b(2), ax.d(8))

    # jax.nn.softmax requires an axis. Calling it on the base tensor should crash
    # with our custom mathematical ambiguity error!
    with pytest.raises(ValueError, match="Mathematical Ambiguity"):
        _ = x.softmax()

    # But targeting the axis first should work perfectly
    safe_out = x.d.softmax()
    assert safe_out.topology == (ax.b(2), ax.d(8))
    print("Mathematical Ambiguity guard successfully triggered!\n")


# ==========================================
# 12. TEST: Dynamic Bundle Routing
# ==========================================
def test_dynamic_bundle_dispatch():
    print("--- Testing Dynamic Bundle Dispatch ---")
    x = init.normal(ax.b(2), ax.d(4))
    y = init.normal(ax.b(2), ax.d(4))

    # 'square' is a JAX primitive, not explicitly defined in Bundle.
    # The Bundle Dispatcher should catch it and route it parallel across both!
    out_x, out_y = (x & y).square()

    assert isinstance(out_x, Tensor)
    assert isinstance(out_y, Tensor)
    assert out_x.topology == (ax.b(2), ax.d(4))

    # Reductions via Bundle targeting
    var_x, var_y = (x & y).d.var()
    assert var_x.topology == (ax.b(2),)
    assert var_y.topology == (ax.b(2),)
    print("Dynamic Bundle Dispatch passed!\n")


# ==========================================
# 13. TEST: Dynamic Axiom NN Module Routing
# ==========================================
def test_dynamic_axiom_nn_routing():
    print("--- Testing Dynamic Axiom NN Routing ---")
    compiler_state.params.clear()
    compiler_state.param_counter = 0

    x = init.normal(ax.b(2), ax.s(4), ax.d(16))

    # layer_norm is defined in axiom.nn and decorated with @axiom_nn_op
    # TargetedTensor should catch it, pass the WHOLE TargetedTensor object to it,
    # and properly initialize the gamma/beta weights via the Ghost Pass.
    out = x.d.layer_norm()

    assert out.topology == (ax.b(2), ax.s(4), ax.d(16))
    assert "test_dynamic_axiom_nn_routing/gamma_0" in compiler_state.params
    assert "test_dynamic_axiom_nn_routing/beta_1" in compiler_state.params
    print("Dynamic Axiom NN routing and parameter allocation passed!\n")


def test_missing_attribute_error():
    print("--- Testing Missing Attribute Fallback ---")
    x = init.normal(ax.d(4))

    # Asking for an axis that doesn't exist or a typo'd function
    with pytest.raises(AttributeError, match="Tensor has no axis, NN function, or JAX primitive 'made_up_func'"):
        _ = x.made_up_func()

    with pytest.raises(AttributeError, match="Targeted axis, NN function, or JAX primitive 'fake_axis' not found"):
        _ = x.d.fake_axis()

    print("Clean AttributeErrors passed!\n")


# ==========================================
# 14. TEST: 1D Spatial Convolution (pad -> unfold -> proj)
# ==========================================
def test_1d_convolution():
    print("--- Testing 1D Convolution Pipeline ---")
    ax.s = ax("s", 5)  # Spatial Sequence
    ax.w = ax("w", 3)  # Window/Kernel Size
    ax.d = ax("d", 4)  # Input Features
    ax.out_d = ax("out_d", 8)  # Output Features

    # Input: [batch=2, seq=5, features=4]
    x = init.normal(ax.b(2), ax.s(5), ax.d(4))

    # 1. Pad sequence: 1 on left, 1 on right (Maintains spatial size after unfold)
    padded = x.s.pad((1, 1))
    assert padded.topology == (ax.b(2), ax.s(7), ax.d(4))

    # 2. Unfold: Slide window of size 3. Out size: (7 - 3)//1 + 1 = 5
    unfolded = padded.s.unfold(ax.w(3), step=1)

    # Topology should now contain the window axis!
    assert unfolded.topology == (ax.b(2), ax.s(5), ax.w(3), ax.d(4))

    # 3. Project: Contract BOTH window and input features into output features
    # This is exactly how a convolution kernel works under the hood!
    out = unfolded.w.d.proj(ax.out_d(8), bias=True)

    assert out.topology == (ax.b(2), ax.s(5), ax.out_d(8))
    print("1D Convolution pipeline passed!\n")


# ==========================================
# 15. TEST: Coordinate Masking (Causal Attention Grid)
# ==========================================
def test_coordinate_masking():
    print("--- Testing Coordinate Masking (Causal TriL) ---")
    ax.q = ax("q", 4)
    ax.k = ax("k", 4)

    # Simulate an unmasked attention matrix [q=4, k=4]
    attn = init.ones(ax.q(4), ax.k(4))

    # Mask out the upper triangle natively using index logic!
    # If q_index < k_index, mask it to -1e9
    causal_attn = attn.q.k.mask(lambda q_idx, k_idx: q_idx < k_idx, fill=-1e9)

    raw = causal_attn.unwrap()

    assert raw[0, 1] == -1e9  # Upper triangle is masked
    assert raw[1, 0] == 1.0  # Lower triangle is untouched
    assert causal_attn.topology == (ax.q(4), ax.k(4))
    print("Coordinate masking passed!\n")


# ==========================================
# 16. TEST: Value Masking (vmask)
# ==========================================
def test_value_masking():
    print("--- Testing Value Masking (vmask) ---")
    ax.d = ax("d", 4)
    x = wrap(jnp.array([0.1, 0.9, 0.4, 0.8]), ax.d)

    # Drop values below 0.5 to zero (ReLU-like behavior)
    dropped = x.vmask(lambda arr: arr < 0.5, fill=0.0)

    assert jnp.allclose(dropped.unwrap(), jnp.array([0.0, 0.9, 0.0, 0.8]))
    assert dropped.topology == (ax.d(4),)
    print("Value masking passed!\n")


# ==========================================
# 17. TEST: Explicit Tie Scope Overrides
# ==========================================
def test_explicit_tie_scope_override():
    print("--- Testing Explicit Tie Scope Override ---")
    compiler_state.params.clear()
    compiler_state.param_counter = 0

    ax.d = ax("d", 4)
    x = init.ones(ax.d)

    # Two separate functions simulating two different model blocks
    def block_a(t): return t.d.bias(tie="@my_global_bias")

    def block_b(t): return t.d.bias(tie="@my_global_bias")

    _ = block_a(x)
    _ = block_b(x)

    # Because they used a global tie, they should NOT register under block_a or block_b
    assert "my_global_bias" in compiler_state.params
    assert len(compiler_state.params) == 1  # They successfully shared the exact same parameter!
    print("Explicit tie overrides passed!\n")


def test_dynamic_axiom_nn_routing():
    print("--- Testing Dynamic Axiom NN Routing ---")
    compiler_state.params.clear()
    compiler_state.reset_pass_state()

    x = init.normal(ax.b(2), ax.s(4), ax.d(16))
    out = x.d.layer_norm()

    assert out.topology == (ax.b(2), ax.s(4), ax.d(16))

    # Notice the _0 prefix! This proves execution isolation works.
    assert "test_dynamic_axiom_nn_routing_0/gamma_0" in compiler_state.params
    assert "test_dynamic_axiom_nn_routing_0/beta_1" in compiler_state.params
    print("Dynamic Axiom NN routing and parameter allocation passed!\n")


def test_1d_convolution():
    print("--- Testing 1D Convolution Pipeline ---")
    compiler_state.params.clear()
    compiler_state.reset_pass_state()

    ax.s = ax("s", 5)  # Spatial Sequence
    ax.w = ax("w", 3)  # Window/Kernel Size
    ax.d = ax("d", 4)  # Input Features
    ax.out_d = ax("out_d", 8)  # Output Features

    # Input: [batch=2, seq=5, features=4]
    x = init.normal(ax.b(2), ax.s(5), ax.d(4))

    # 1. Pad sequence: 1 on left, 1 on right (Maintains spatial size after unfold)
    padded = x.s.pad((1, 1))
    assert padded.topology == (ax.b(2), ax.s(7), ax.d(4))

    # 2. Unfold: Slide window of size 3. Out size: (7 - 3)//1 + 1 = 5
    unfolded = padded.s.unfold(ax.w(3), step=1)

    # Topology should now contain the window axis!
    assert unfolded.topology == (ax.b(2), ax.s(5), ax.w(3), ax.d(4))

    # 3. Project: Contract BOTH window and input features into output features!
    out = unfolded.w.d.proj(ax.out_d(8), bias=True)

    assert out.topology == (ax.b(2), ax.s(5), ax.out_d(8))
    assert "test_1d_convolution_0/proj_w_0" in compiler_state.params
    print("1D Convolution pipeline passed!\n")


def test_coordinate_masking():
    print("--- Testing Coordinate Masking (Causal TriL) ---")
    ax.q = ax("q", 4)
    ax.k = ax("k", 4)

    attn = init.ones(ax.q(4), ax.k(4))

    # Mask out the upper triangle natively using index logic!
    # If q_index < k_index, mask it to -1e9
    causal_attn = attn.q.k.mask(lambda q_idx, k_idx: q_idx < k_idx, fill=-1e9)
    raw = causal_attn.unwrap()

    assert raw[0, 1] == -1e9  # Upper triangle is masked
    assert raw[1, 0] == 1.0  # Lower triangle is untouched
    assert causal_attn.topology == (ax.q(4), ax.k(4))
    print("Coordinate masking passed!\n")


def test_explicit_tie_scope_override():
    print("--- Testing Explicit Tie Scope Override ---")
    compiler_state.params.clear()
    compiler_state.reset_pass_state()

    ax.d = ax("d", 4)
    x = init.ones(ax.d)

    # Two separate functions simulating two different model blocks
    def block_a(t): return t.d.bias(tie="@my_global_bias")

    def block_b(t): return t.d.bias(tie="@my_global_bias")

    _ = block_a(x)
    _ = block_b(x)

    # Because they used a global tie, they should NOT register under block_a or block_b
    assert "my_global_bias" in compiler_state.params
    assert len(compiler_state.params) == 1  # They successfully shared the exact same parameter!
    print("Explicit tie overrides passed!\n")


# ==========================================
# 18. TEST: Single Tensor Merging
# ==========================================
def test_tensor_merge():
    print("--- Testing Tensor Axis Merging ---")
    ax.b = ax("b", 2)
    ax.h = ax("h", 8)
    ax.w = ax("w", 8)
    ax.d = ax("d", 16)

    # 1. Standard Merge
    x = init.normal(ax.b(2), ax.h(8), ax.w(8), ax.d(16))
    ax.spatial = ax("spatial", 64)  # 8 * 8 = 64

    merged = x.h.w.merge(ax.spatial)

    # Topology should reflect the merged axis at the end!
    assert merged.topology == (ax.b(2), ax.d(16), ax.spatial(64))
    assert merged.unwrap().shape == (2, 16, 64)

    # 2. Strict Size Guard Catch
    ax.bad_spatial = ax("bad_spatial", 63)
    with pytest.raises(ValueError, match="Topological Violation"):
        _ = x.h.w.merge(ax.bad_spatial)

    print("Single Tensor merge passed!\n")


# ==========================================
# 19. TEST: Parallel Bundle Merging
# ==========================================
def test_bundle_merge():
    print("--- Testing Parallel Bundle Merging ---")
    ax.b = ax("b", 2)
    ax.heads = ax("heads", 4)
    ax.seq = ax("seq", 128)
    ax.head_dim = ax("head_dim", 64)

    # Simulate multi-head attention outputs
    q = init.normal(ax.b, ax.heads, ax.seq, ax.head_dim)
    k = init.normal(ax.b, ax.heads, ax.seq, ax.head_dim)
    v = init.normal(ax.b, ax.heads, ax.seq, ax.head_dim)

    # We want to merge `heads` and `head_dim` into `d_model` (4 * 64 = 256)
    ax.d_model = ax("d_model", 256)

    # Parallel Merge!
    merged_q, merged_k, merged_v = (q & k & v).heads.head_dim.merge(ax.d_model)

    # Verify they were all merged correctly
    for t in (merged_q, merged_k, merged_v):
        assert t.topology == (ax.b(2), ax.seq(128), ax.d_model(256))

    # Verify infinite chaining still works on merged bundles
    chained_proj = (q & k & v).heads.head_dim.merge(ax.d_model).d_model.proj(ax("out", 128))
    for t in chained_proj:
        assert t.topology == (ax.b(2), ax.seq(128), ax("out", 128))

    print("Parallel Bundle merge passed!\n")


# ==========================================
# 20. TEST: Bundle Stacking (GQA Pattern)
# ==========================================
def test_bundle_stacking():
    print("--- Testing Bundle Stacking (GQA Pattern) ---")
    ax.b = ax("b", 2)
    ax.seq = ax("seq", 10)
    ax.d = ax("d", 64)

    # Simulate a generic input state
    x = init.normal(ax.b, ax.seq, ax.d)

    # GQA Config
    kv_heads = 2
    q_heads = 8
    head_dim = 64 // (kv_heads + q_heads)  # 64 // 10 = 6 (just for integer testing)

    ax.kvh = ax("kvh", kv_heads)
    ax.qh = ax("qh", q_heads)
    ax.h = ax("h", head_dim)

    # 1. Project into Keys and Values
    k, v = (x & x).d.proj(ax.kvh, ax.h, bias=False)

    # 2. Replicate K/V for the Query Heads via Bundle Stacking!
    stacked_bundle = ax.stack([k & v for _ in range(q_heads // kv_heads)], ax("qh_group", q_heads // kv_heads))

    assert isinstance(stacked_bundle, Bundle)
    # The new stack axis should be injected at the front of the topology
    assert stacked_bundle.tensors[0].topology == (ax("qh_group", 4), ax.b, ax.seq, ax.kvh, ax.h)

    # 3. Merge the parallel stack groups with the KV heads to match the Q heads
    k_merged, v_merged = stacked_bundle.qh_group.kvh.merge(ax.qh)

    assert k_merged.topology == (ax.b, ax.seq, ax.h, ax.qh(8))
    print("Bundle stacking passed!\n")


# ==========================================
# 21. TEST: Pure JAX Conversion and Manual VJP
# ==========================================
def test_to_jax_and_manual_vjp():
    print("--- Testing to_jax and Manual VJP ---")
    ax.b = ax("b", 2)
    ax.d = ax("d", 4)

    # 1. Define a standard Axiom model
    def my_net(x):
        return x.d.proj(ax("out", 8), bias=True).out.gelu()

    model = ax.model(my_net)

    # 2. Initialize it via ghost pass
    x = init.normal(ax.b, ax.d)
    _ = model(x)

    # 3. Strip it down to pure JAX!
    from axiom.compiler import to_jax
    params, apply_fn = to_jax(model)

    assert isinstance(params, dict)
    assert "my_net_0/proj_w_0" in params

    # 4. Perform a manual JAX Vector-Jacobian Product (VJP)
    import jax

    # We want gradients with respect to params and inputs
    def fwd(p, inputs):
        return apply_fn(p, inputs)

    # Get the outputs and the backward-pass function
    primals_out, vjp_fn = jax.vjp(fwd, params, x)

    assert primals_out.topology == (ax.b(2), ax("out", 8))

    # 5. Push a cotangent vector backward!
    # Cotangents must match the shape of the output
    cotangent = init.ones(ax.b(2), ax("out", 8))

    grad_params, grad_inputs = vjp_fn(cotangent)

    # Validate the gradients were successfully computed
    assert "my_net_0/proj_w_0" in grad_params
    assert grad_inputs.topology == (ax.b(2), ax.d(4))

    print("to_jax and manual VJP passed!\n")


# ==========================================
# 22. TEST: Topological Splitting
# ==========================================
def test_tensor_split():
    print("--- Testing Tensor Axis Splitting ---")
    ax.b = ax("b", 2)
    ax.d = ax("d", 32)
    x = init.normal(ax.b, ax.d)

    # 1. Split with explicit sizes
    ax.heads = ax("heads", 4)
    ax.h_dim = ax("h_dim", 8)
    split_explicit = x.d.split(ax.heads, ax.h_dim)

    assert split_explicit.topology == (ax.b(2), ax.heads(4), ax.h_dim(8))

    # 2. Split with INFERRED size!
    ax.heads_inf = ax("heads_inf")  # Notice we pass NO size here!
    split_inferred = x.d.split(ax.heads_inf, ax.h_dim(8))

    # Axiom should perfectly calculate that 32 // 8 = 4
    assert split_inferred.topology == (ax.b(2), ax("heads_inf", 4), ax.h_dim(8))

    # 3. Guard: Invalid inference math
    with pytest.raises(ValueError, match="cleanly divide"):
        _ = x.d.split(ax("bad_inf"), ax("bad_dim", 7))

    print("Single Tensor split passed!\n")


# ==========================================
# 23. TEST: Parallel Bundle Splitting
# ==========================================
def test_bundle_split():
    print("--- Testing Parallel Bundle Splitting ---")
    ax.b = ax("b", 2)
    ax.d = ax("d", 32)

    q = init.normal(ax.b, ax.d)
    k = init.normal(ax.b, ax.d)
    v = init.normal(ax.b, ax.d)

    # Split them all in parallel, inferring the head dimension!
    q_split, k_split, v_split = (q & k & v).d.split(ax("heads", 4), ax("h_dim"))

    for t in (q_split, k_split, v_split):
        assert t.topology == (ax.b(2), ax("heads", 4), ax("h_dim", 8))

    print("Bundle splitting passed!\n")


# ==========================================
# 24. TEST: init.arange primitive
# ==========================================
def test_init_arange():
    print("--- Testing init.arange ---")

    # 1. Standard count
    # Use ax("s") instead of ax.s to prevent state bleed from previous tests!
    t = init.arange(10, ax("s"))
    assert t.topology == (ax("s", 10),)
    assert t.unwrap()[9] == 9

    # 2. Stepped count with strict axis
    # Here we explicitly declare the sized axis locally
    strict_ax = ax("hd", 5)
    t_step = init.arange(0, 10, 2, strict_ax)
    assert t_step.topology == (strict_ax,)
    assert t_step.unwrap()[4] == 8

    print("init.arange passed!\n")


# ==========================================
# 24. TEST: Targeted Tensor Unpacking
# ==========================================
def test_tensor_unpacking():
    print("--- Testing Targeted Tensor Unpacking ---")

    # Create a simple predictable array: shape (2, 3)
    raw = jnp.array([[1., 2., 3.],
                     [4., 5., 6.]])
    x = Tensor(raw, ax.b(2), ax.s(3))

    # Pythonic unpacking across the sequence axis!
    x0, x1, x2 = x.s

    # 1. Topological Safety: The 's' axis must be completely dropped
    expected_top = (ax.b(2),)
    assert x0.topology == expected_top
    assert x1.topology == expected_top
    assert x2.topology == expected_top

    # 2. Mathematical Precision
    assert jnp.allclose(x0.unwrap(), jnp.array([1., 4.]))
    assert jnp.allclose(x1.unwrap(), jnp.array([2., 5.]))
    assert jnp.allclose(x2.unwrap(), jnp.array([3., 6.]))

    print("Targeted Tensor unpacking passed!\n")


# ==========================================
# 25. TEST: Parallel Bundle Unpacking
# ==========================================
def test_bundle_unpacking():
    print("--- Testing Parallel Bundle Unpacking ---")

    # Create mock Query and Key matrices
    q = init.ones(ax.b(2), ax.h(2), ax.d(4))
    k = init.zeros(ax.b(2), ax.h(2), ax.d(4))

    # Pythonic parallel unpacking across the head axis!
    (q0, k0), (q1, k1) = (q & k).h

    # 1. Topological Safety: The 'h' axis must be dropped for all tensors
    expected_top = (ax.b(2), ax.d(4))
    assert q0.topology == expected_top
    assert k0.topology == expected_top
    assert q1.topology == expected_top
    assert k1.topology == expected_top

    # 2. Mathematical Precision
    assert jnp.allclose(q0.unwrap(), jnp.ones((2, 4)))
    assert jnp.allclose(k1.unwrap(), jnp.zeros((2, 4)))

    print("Parallel Bundle unpacking passed!\n")


# ==========================================
# 26. TEST: Recurrent Weight Tying (.repeat)
# ==========================================
def test_repeat_weight_tying():
    print("--- Testing Recurrent Weight Tying (.repeat) ---")

    # 1. Clear global compiler state to prevent bleed from previous tests
    compiler_state.params.clear()
    compiler_state.reset_pass_state()

    # ----------------------------------------
    # Part 1: Single Tensor Repeat
    # ----------------------------------------
    def recurrent_block(x: Tensor) -> Tensor:
        # THE FIX: Use ax("d") instead of the globally polluted ax.d
        # Your new inference engine will perfectly infer its size as 4!
        return x.d.proj(ax("d"))

    # Explicitly instantiate the sized axes to avoid global bleed
    x = init.ones(ax("b", 2), ax("d", 4))

    # Execute the block 5 times
    out = x.repeat(recurrent_block, times=5)

    # Topological Safety
    assert out.topology == (ax("b", 2), ax("d", 4)), "Topology mismatch after repeat."

    # Weight Sharing Proof
    # 1 weight + 1 bias = 2 parameters total for the entire recurrent loop!
    assert len(compiler_state.params) == 2, f"Weight tying failed! Params: {compiler_state.params.keys()}"

    print("Tensor .repeat() weight tying passed!")

    # ----------------------------------------
    # Part 2: Parallel Bundle Repeat
    # ----------------------------------------
    compiler_state.params.clear()
    compiler_state.reset_pass_state()

    def bundle_block(b):
        return b.d.proj(ax("d"))

    b_in = x & x

    # Execute the parallel bundle block 5 times
    b_out = b_in.repeat(bundle_block, times=5)

    # Weight Sharing Proof for Bundles
    # 2 tensors * (1 weight + 1 bias) = 4 parameters total.
    assert len(compiler_state.params) == 4, f"Bundle tying failed! Params: {compiler_state.params.keys()}"

    print("Bundle .repeat() weight tying passed!\n")


def test_rope_partial_rotation():
    print("--- Testing Partial Rotary Position Embedding ---")
    import jax.numpy as jnp
    from axiom import ax, wrap

    ax.s = ax("s", 4)
    ax.d = ax("d", 16)

    # Create a deterministic sequence tensor
    raw_x = jnp.arange(4 * 16, dtype=jnp.float32).reshape(4, 16)
    x = wrap(raw_x, ax.s, ax.d)

    # Apply 50% partial RoPE!
    # Notice how we can call .rope() directly thanks to your dynamic __getattr__!
    out = x.d.rope(seq_ax=ax.s, rot_fraction=0.5)

    # 1. Topological Safety
    assert out.topology == x.topology, "RoPE illegally altered the topology!"

    # 2. Functional Safety: The unrotated half MUST be perfectly identical
    # Because rot_fraction=0.5 and d=16, indices 8:16 should be untouched.
    x_unrotated = x.d[8:].unwrap()
    out_unrotated = out.d[8:].unwrap()
    assert jnp.allclose(x_unrotated, out_unrotated), "Partial RoPE corrupted the pass-through dimensions!"

    # 3. Mathematical Execution: The rotated half MUST change
    x_rotated = x.d[:8].unwrap()
    out_rotated = out.d[:8].unwrap()
    assert not jnp.allclose(x_rotated, out_rotated), "RoPE failed to rotate the targeted dimensions!"


def test_infonce_contrastive_matching():
    print("--- Testing InfoNCE Contrastive Loss ---")
    import jax.numpy as jnp
    from axiom import ax, wrap, nn

    ax.b = ax("b", 2)
    ax.d = ax("d", 2)

    # Queries: A batch of two distinct one-hot vectors
    query = wrap(jnp.array([[1.0, 0.0], [0.0, 1.0]]), ax.b, ax.d)

    # Perfect Keys: Exact match to the queries
    k_perfect = wrap(jnp.array([[1.0, 0.0], [0.0, 1.0]]), ax.b, ax.d)

    # Opposite Keys: Completely mismatched
    k_opposite = wrap(jnp.array([[0.0, 1.0], [1.0, 0.0]]), ax.b, ax.d)

    # Compute both losses using our new topological reduction default
    loss_perfect = nn.infonce_loss(query, k_perfect, batch_ax=ax.b, temp=0.1)
    loss_opposite = nn.infonce_loss(query, k_opposite, batch_ax=ax.b, temp=0.1)

    # 1. Topological Safety
    assert len(loss_perfect.topology) == 0, "InfoNCE default reduction failed to collapse to a scalar!"

    # 2. Mathematical Safety: Mismatched keys should result in a higher loss!
    assert loss_perfect.item() < loss_opposite.item(), "InfoNCE failed to penalize mismatched contrastive pairs!"