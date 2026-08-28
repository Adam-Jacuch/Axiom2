import jax
import jax.numpy as jnp
import pytest

from axiom import AxisStream, Tensor, TiledAxisRef, ax, wrap


def test_tiled_map_exposes_named_grid_and_register_indices():
    x = wrap(jnp.arange(16, dtype=jnp.float32), ax.s(16))
    tiled = x.s(4)

    assert isinstance(tiled, TiledAxisRef)
    assert isinstance(tiled, AxisStream)

    def add_grid_coordinates(block):
        return Tensor(block.unwrap() + ax.grid[ax.s] + ax.tile[ax.s], *block.topology)

    actual = tiled.map(add_grid_coordinates)
    expected = jnp.array([0, 2, 4, 6, 5, 7, 9, 11, 10, 12, 14, 16, 15, 17, 19, 21], dtype=jnp.float32)

    assert actual.topology == (ax.s(16),)
    assert jnp.array_equal(actual.unwrap(), expected)


def test_bundled_blocked_matmul_matches_jax_matmul():
    m, k, n = ax.m(4), ax.k(5), ax.n(6)
    left = wrap(jnp.arange(m.size * k.size, dtype=jnp.float32).reshape(m.size, k.size), m, k)
    right = wrap(jnp.arange(k.size * n.size, dtype=jnp.float32).reshape(k.size, n.size), k, n)

    def matmul_tile(inputs):
        left_tile, right_tile = inputs
        return Tensor(left_tile.unwrap() @ right_tile.unwrap(), ax.m(2), ax.n(3))

    actual = (left & right).m(2).n(3).map(matmul_tile)

    assert actual.topology == (m, n)
    assert jnp.allclose(actual.unwrap(), left.unwrap() @ right.unwrap())


def test_fused_flash_attention_fold_matches_eager_attention():
    b, h, q, k, d = ax.b(1), ax.h(1), ax.q(4), ax.k(4), ax.d(3)
    q_raw, k_raw, v_raw = jax.random.normal(jax.random.key(0), (3, b.size, h.size, q.size, d.size))
    query = wrap(q_raw, b, h, q, d)
    key = wrap(k_raw, b, h, k, d)
    value = wrap(v_raw, b, h, k, d)

    def flash_tile(inputs):
        q_tile, k_full, v_full = inputs
        q_values = q_tile.unwrap()
        init = (
            jnp.full(q_values.shape[:-1], -jnp.inf, dtype=q_values.dtype),
            jnp.zeros(q_values.shape[:-1], dtype=q_values.dtype),
            jnp.zeros_like(q_values),
        )

        def online_softmax_step(carry, kv_tile):
            max_so_far, normalizer, accumulator = carry
            k_tile, v_tile = kv_tile
            scores = jnp.einsum("bhqd,bhkd->bhqk", q_values, k_tile.unwrap()) / jnp.sqrt(d.size)
            next_max = jnp.maximum(max_so_far, jnp.max(scores, axis=-1))
            previous_weight = jnp.exp(max_so_far - next_max)
            weights = jnp.exp(scores - next_max[..., None])
            return (
                next_max,
                previous_weight * normalizer + jnp.sum(weights, axis=-1),
                previous_weight[..., None] * accumulator
                + jnp.einsum("bhqk,bhkd->bhqd", weights, v_tile.unwrap()),
            )

        _, normalizer, accumulator = (k_full & v_full).k(2).fold(
            online_softmax_step, init=init, stages=2
        )
        return Tensor(accumulator / normalizer[..., None], *q_tile.topology)

    actual = (query & key & value).b(1).h(1).q(2).map(flash_tile)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q_raw, k_raw) / jnp.sqrt(d.size)
    expected = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, axis=-1), v_raw)

    assert jnp.allclose(actual.unwrap(), expected, atol=1e-5, rtol=1e-5)


def test_fold_accepts_dynamic_until_and_stages():
    x = wrap(jnp.arange(8, dtype=jnp.float32).reshape(2, 4), ax.b(2), ax.s(4))

    def body(block):
        carry = block.s(2).fold(
            lambda total, subtile: total + jnp.sum(subtile.unwrap(), axis=1),
            init=jnp.zeros((1,), dtype=jnp.float32),
            until=ax.grid[ax.b] + 1,
            stages=2,
        )
        return Tensor(carry, ax.b(1))

    actual = x.b(1).map(body)

    assert jnp.array_equal(actual.unwrap(), jnp.array([1.0, 22.0]))


def test_tiled_map_grad_and_remat_are_composable():
    def loss(raw):
        tiled = wrap(raw, ax.s(8)).s(4).map(lambda block: block * 2.0)
        return jnp.sum(tiled.unwrap())

    values = jnp.arange(8, dtype=jnp.float32)

    assert jnp.array_equal(jax.grad(loss)(values), jnp.full_like(values, 2.0))
    assert jnp.array_equal(jax.grad(ax.remat(loss))(values), jnp.full_like(values, 2.0))


def test_non_divisible_axis_boundary_padding():
    x = wrap(jnp.ones((14,), dtype=jnp.float32), ax.s(14))

    actual = x.s(4).map(lambda block: block * 2.0)

    def loss(raw):
        return jnp.sum(wrap(raw, ax.s(14)).s(4).map(lambda block: block * 2.0).unwrap())

    assert actual.unwrap().shape == (14,)
    assert jnp.allclose(actual.unwrap(), 2.0)
    assert jnp.array_equal(jax.grad(loss)(x.unwrap()), jnp.full((14,), 2.0))


def test_mixed_precision_fold_accumulates_in_float32_then_casts_back():
    x = wrap(jnp.ones((8, 8), dtype=jnp.bfloat16), ax.m(8), ax.k(8))

    def reduce_k(block):
        accumulator = block.k(2).fold(
            lambda carry, k_tile: carry + jnp.sum(k_tile.unwrap().astype(jnp.float32), axis=1),
            init=jnp.zeros((8,), dtype=jnp.float32),
            stages=2,
        )
        return Tensor(accumulator.astype(jnp.bfloat16), ax.m(8))

    actual = x.m(8).map(reduce_k)

    assert actual.unwrap().dtype == jnp.bfloat16
    assert jnp.array_equal(actual.unwrap(), jnp.full((8,), 8, dtype=jnp.bfloat16))


def test_tiled_axes_can_surround_an_untiled_intermediate_axis():
    x = wrap(
        jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8),
        ax.b(2),
        ax.heads(3),
        ax.d(8),
    )

    actual = x.b(1).d(4).map(lambda block: block * 3.0)

    assert actual.topology == x.topology
    assert jnp.array_equal(actual.unwrap(), x.unwrap() * 3.0)


def test_cpu_uses_pallas_interpret_fallback(monkeypatch):
    if jax.devices()[0].platform != "cpu":
        pytest.skip("CPU-only fallback check")

    import axiom.kernel as kernel

    observed = {}
    original = kernel.pl.pallas_call

    def spy(*args, **kwargs):
        observed["interpret"] = kwargs["interpret"]
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel.pl, "pallas_call", spy)
    actual = wrap(jnp.ones((8,), dtype=jnp.float32), ax.s(8)).s(4).map(lambda block: block + 1)

    assert observed["interpret"] is True
    assert jnp.allclose(actual.unwrap(), 2.0)


def test_accelerator_uses_native_pallas_lowering(monkeypatch):
    if jax.devices()[0].platform not in {"gpu", "tpu"}:
        pytest.skip("Requires a GPU or TPU Pallas backend")

    import axiom.kernel as kernel

    observed = {}
    original = kernel.pl.pallas_call

    def spy(*args, **kwargs):
        observed["interpret"] = kwargs["interpret"]
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel.pl, "pallas_call", spy)
    x = wrap(jnp.ones((256,), dtype=jnp.float32), ax.s(256))
    actual = x.s(128).map(lambda block: block + 1.0)
    actual.unwrap().block_until_ready()

    assert observed["interpret"] is False
    assert jnp.allclose(actual.unwrap(), 2.0)


def test_map_returning_multiple_tensors():
    x = wrap(jnp.arange(16, dtype=jnp.float32), ax.s(16))

    def split_block(b):
        return (b * 2.0) & (b + 10.0)

    out1, out2 = x.s(4).map(split_block)
    assert jnp.array_equal(out1.unwrap(), x.unwrap() * 2.0)
    assert jnp.array_equal(out2.unwrap(), x.unwrap() + 10.0)


def test_fold_zero_bound_edge_case():
    x = wrap(jnp.ones((4,)), ax.s(4))
    res = x.s(2).fold(lambda c, b: c + 1, init=0, until=0)
    assert int(res) == 0
