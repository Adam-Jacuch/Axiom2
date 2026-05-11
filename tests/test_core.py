import pytest
import jax.numpy as jnp
import jax
from axiom import ax, Tensor, wrap


def test_axis_and_tensor():
    ax.b = ax("b", 32)  # Changed from "batch"
    ax.d = ax("d", 512)  # Changed from "dim"

    raw_array = jnp.ones((32, 512))
    x = wrap(raw_array, ax.b, ax.d)

    assert x.topology == (ax.b, ax.d)


def test_pointwise_chaining():
    ax.b = ax("b", 32)
    ax.d = ax("d", 512)

    raw_array = jnp.ones((32, 512))
    x = wrap(raw_array, ax.b, ax.d)

    # Now x.d will correctly find the axis named "d"
    x_act = x.d.pw(jax.nn.sigmoid)

    assert isinstance(x_act, Tensor)
    assert x_act.topology == (ax.b, ax.d)
    assert jnp.allclose(x_act.unwrap(), jax.nn.sigmoid(jnp.ones((32, 512))))


def test_projection_topology_shift():
    ax.b = ax("b", 32)
    ax.d = ax("d", 512)
    ax.d2 = ax("d2", 1024)

    x = wrap(jnp.ones((32, 512)), ax.b, ax.d)

    # Project 'd' (512) to 'd2' (1024)
    x_proj = x.d.proj(ax.d2)

    # Check that the old axis is gone and the new one is present!
    assert x_proj.topology == (ax.b, ax.d2)
    assert x_proj.unwrap().shape == (32, 1024)

    # Check fluent chaining with the new axis
    x_chained = x.d.proj(ax.d2).d2.pw(jax.nn.relu)
    assert x_chained.topology == (ax.b, ax.d2)


def test_native_broadcasting():
    ax.b = ax("b", 32)
    ax.s = ax("s", 1024)
    ax.s_k = ax("s_k", 512)

    # Tensor A: [b, s]
    A = wrap(jnp.ones((32, 1024)), ax.b, ax.s)

    # Tensor B: [b, s_k]
    B = wrap(jnp.ones((32, 512)), ax.b, ax.s_k)

    # A + B should automatically expand to [b, s, s_k]
    C = A + B

    assert C.topology == (ax.b, ax.s, ax.s_k)
    assert C.unwrap().shape == (32, 1024, 512)

    # Test scalar math
    C_scaled = C * 2.0
    assert jnp.allclose(C_scaled.unwrap(), jnp.full((32, 1024, 512), 4.0))


def test_topological_masking():
    ax.s = ax("s", 4)
    ax.s_k = ax("s_k", 4)

    # A tiny sequence attention grid [s, s_k]
    scores = wrap(jnp.zeros((4, 4)), ax.s, ax.s_k)

    # Apply a causal mask! (s >= s_k)
    masked_scores = scores.s.s_k.mask(lambda s, s_k: s >= s_k, fill=-1e9)

    assert masked_scores.topology == (ax.s, ax.s_k)

    # Check the actual values (Upper right triangle should be -1e9)
    raw_result = masked_scores.unwrap()
    assert raw_result[0, 1] == -1e9  # s=0, s_k=1 (0 >= 1 is False, masked)
    assert raw_result[1, 0] == 0.0  # s=1, s_k=0 (1 >= 0 is True, kept)


def test_param_tagging():
    ax.d = ax("d", 128)
    W = wrap(jnp.ones((128,)), ax.d).param(name="embedding_weight")

    assert W._is_param is True
    assert W._param_name == "embedding_weight"


def test_reductions():
    ax.b = ax("b", 32)
    ax.s = ax("s", 1024)
    ax.d = ax("d", 512)

    x = wrap(jnp.ones((32, 1024, 512)), ax.b, ax.s, ax.d)

    # Pillar IV: Reduce over the sequence axis
    sentence_embeddings = x.s.mean()

    # The 's' axis should be permanently removed from the topology
    assert sentence_embeddings.topology == (ax.b, ax.d)
    assert sentence_embeddings.unwrap().shape == (32, 512)

    # Reduce over two axes at once! (Batch and Dim)
    global_scalar = x.b.d.sum()
    assert global_scalar.topology == (ax.s,)
    assert global_scalar.unwrap().shape == (1024,)


def test_bundle_functor_projections():
    ax.b = ax("b", 16)
    ax.d = ax("d", 256)
    ax.heads = ax("heads", 8)
    ax.hd = ax("hd", 32)

    # Create Q, K, V dummy inputs
    x = wrap(jnp.ones((16, 256)), ax.b, ax.d)
    y = wrap(jnp.ones((16, 256)), ax.b, ax.d)
    z = wrap(jnp.ones((16, 256)), ax.b, ax.d)

    # THE MAGIC: Project all three tensors in parallel using the & operator!
    q, k, v = (x & y & z).d.proj(ax.heads, ax.hd)

    # Verify all three outputs mapped to the new topology successfully
    assert q.topology == (ax.b, ax.heads, ax.hd)
    assert k.topology == (ax.b, ax.heads, ax.hd)
    assert v.topology == (ax.b, ax.heads, ax.hd)

    assert q.unwrap().shape == (16, 8, 32)


def test_standard_scan():
    ax.b = ax("b", 16)
    ax.s = ax("s", 128)
    ax.d = ax("d", 64)

    # Input sequence [b, s, d]
    x = wrap(jnp.ones((16, 128, 64)), ax.b, ax.s, ax.d)

    # Initial hidden state [b, d]
    h_init = wrap(jnp.zeros((16, 64)), ax.b, ax.d)

    def simple_rnn(carry, xt):
        # Axiom native addition handles the [b, d] logic perfectly
        new_carry = carry + xt
        return new_carry, new_carry

    final_h, h_seq = x.s.scan(simple_rnn, init=h_init)

    assert final_h.topology == (ax.b, ax.d)
    assert h_seq.topology == (ax.b, ax.s, ax.d)

    # The last element of the cumulative sum of 1s over 128 steps should be 128.0
    assert jnp.allclose(final_h.unwrap(), 128.0)


def test_associative_scan_ssm():
    ax.b = ax("b", 8)
    ax.s = ax("s", 256)
    ax.d = ax("d", 32)

    # A defines transition, B defines input
    A = wrap(jnp.full((8, 256, 32), 0.5), ax.b, ax.s, ax.d)
    B = wrap(jnp.ones((8, 256, 32)), ax.b, ax.s, ax.d)

    def ssm_binary_operator(left, right):
        A_i, B_i = left
        A_j, B_j = right
        # Native Axiom broadcasting guarantees mathematical purity here
        return (A_j * A_i), ((A_j * B_i) + B_j)

    # Parallelize the recurrence!
    A_cum, hidden_states = (A & B).s.assoc_scan(ssm_binary_operator)

    assert A_cum.topology == (ax.b, ax.s, ax.d)
    assert hidden_states.topology == (ax.b, ax.s, ax.d)

    # Verify the associative scan executed correctly
    raw_h = hidden_states.unwrap()
    assert raw_h.shape == (8, 256, 32)


def test_context_routing_gather():
    import jax

    ax.b = ax("b", 4)
    ax.s = ax("s", 128)
    ax.vocab = ax("vocab", 50257)
    ax.d = ax("d", 512)

    # 1. Dummy Tokens [batch, seq]
    key = jax.random.PRNGKey(0)
    tokens_raw = jax.random.randint(key, (4, 128), 0, 50257)
    tokens = wrap(tokens_raw, ax.b, ax.s)

    # 2. Dummy Embedding Matrix [vocab, dim]
    embeddings = wrap(jnp.ones((50257, 512)), ax.vocab, ax.d)

    # THE MAGIC: Target 'vocab', slice by 'tokens', gather, and pop the context!
    embedded = embeddings.vocab[tokens].gather()

    # The 'vocab' axis is gone, replaced perfectly by the [b, s] token topology!
    assert embedded.topology == (ax.b, ax.s, ax.d)
    assert embedded.unwrap().shape == (4, 128, 512)
