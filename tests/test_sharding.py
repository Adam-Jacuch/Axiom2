"""Integration coverage for explicit named-axis device placement.

Run this module in isolation: it requests four CPU devices before importing
JAX, exactly like a small dp=2/tp=2 accelerator mesh.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from axiom import Tensor, ax, init


def _mesh():
    if jax.device_count() != 4:
        pytest.skip("This module must be run before another test imports JAX; run tests/test_sharding.py in isolation.")
    return ax.mesh(dp=2, tp=2)


def _projection_model(mesh):
    b, d, h = ax.b[mesh.dp](8), ax.d(4), ax.h[mesh.tp](6)

    @ax.remat
    def block(x):
        return x.d.proj(h)

    return ax.model(block, mesh=mesh).init(b, d), b, d, h


def test_axis_placement_is_immutable_and_composes_with_normal_slicing():
    mesh = _mesh()
    x = init.ones(ax.b[mesh.dp](2), ax.d[mesh.tp](8))

    placed = x.d[mesh.tp]
    sliced = placed[:3]

    assert placed.topology[1].placement == mesh.tp
    assert sliced.topology == (ax.b[mesh.dp](2), ax.d[mesh.tp](3))
    assert sliced.unwrap().shape == (2, 3)

    with pytest.raises(ValueError, match="already placed"):
        _ = x.d[mesh.dp]


def test_export_selects_legal_tp_projection_specs_and_native_jax_apply():
    mesh = _mesh()
    model, b, d, h = _projection_model(mesh)
    params, apply_fn, layout = ax.to_jax(model, sharding=True)
    specs = layout.explain(params)

    # A column-parallel projection is legal even though its result is TP
    # sharded: the weight never receives the invalid P("tp", "tp") spec.
    weight_spec = next(spec for name, spec in specs.items() if "proj_w" in name)
    bias_spec = next(spec for name, spec in specs.items() if "bias" in name)
    assert weight_spec == PartitionSpec(None, "tp")
    assert bias_spec == PartitionSpec("tp",)
    assert all(isinstance(value.sharding, NamedSharding) for value in params.values())

    x = init.ones(b, d)

    def native_apply(native_params, raw_x):
        return apply_fn(native_params, Tensor(raw_x, b, d)).unwrap()

    with mesh.jax_mesh:
        compiled = jax.jit(
            native_apply,
            in_shardings=(layout.parameter_shardings(params), layout.tensor_sharding(x)),
            out_shardings=layout.tensor_sharding(init.ones(b, h)),
        )
        actual = compiled(params, x.unwrap())

    assert actual.shape == (b.size, h.size)
    assert actual.sharding.spec == PartitionSpec("dp", "tp")


def test_none_explicitly_requests_replication_and_selects_a_row_parallel_weight():
    mesh = _mesh()
    b, d = ax.b[mesh.dp](8), ax.d[mesh.tp](4)

    def row_parallel(x):
        return x.d[mesh.tp].proj(ax.d[None])

    model = ax.model(row_parallel, mesh=mesh).init(b, d)
    params, _, layout = ax.to_jax(model, sharding=True)
    weight_spec = next(spec for name, spec in layout.explain(params).items() if "proj_w" in name)

    assert weight_spec == PartitionSpec("tp", None)
    assert model(init.ones(b, d)).topology[-1].replicated
    assert init.ones(b, d).d[None].topology[-1].replicated


def test_jit_grad_optimizer_state_and_remat_keep_layouts():
    mesh = _mesh()
    model, b, d, _ = _projection_model(mesh)
    x = init.ones(b, d)

    @ax.jit(mesh=mesh)
    def forward(m, inputs):
        return m(inputs)

    out = forward(model, x)
    assert out.unwrap().sharding.spec == PartitionSpec("dp", "tp")

    def loss(m):
        return m(x).h.sum().b.sum()

    grads = ax.grad(loss)(model)
    optimizer = optax.adam(1e-3)
    updated, state = ax.apply_updates(model, grads, optimizer, None)

    for name, value in updated.params.items():
        assert value.sharding.spec == updated.layout.parameter_specs(updated.params)[name]
    state_shardings = updated.layout.state_shardings(state, updated.params)
    state_leaves = jax.tree_util.tree_leaves(state_shardings)
    assert state_leaves and all(isinstance(sharding, NamedSharding) for sharding in state_leaves)

    scaled = updated * 0.5
    assert scaled.mesh is mesh
    assert scaled.layout.explain(scaled.params) == updated.layout.explain(updated.params)


def test_remat_pallas_export_grad_and_checkpoint_keep_one_stable_scope(tmp_path):
    """Exercise the exact composition used by a sharded FlashAttention block.

    The remat body is invoked twice, contains a Pallas map, exports to native
    JAX, and is differentiated under an explicit mesh.  This used to be where
    checkpoint tracing could rediscover a different Python stack scope and
    request parameter names that did not exist in the initialized model.
    """
    mesh = _mesh()
    b, s, d, h = ax.b[mesh.dp](8), ax.s(8), ax.d(4), ax.h[mesh.tp](4)

    @ax.remat
    def block(values):
        projected = values.d.proj(h)
        projected = projected.s(4).map(lambda tile: tile * 2.0, interpret=True)
        return projected.h.proj(d[None])

    def stacked_blocks(values):
        return block(block(values))

    model = ax.model(stacked_blocks, mesh=mesh).init(b, s, d)
    params, apply_fn, layout = ax.to_jax(model, sharding=True)
    values = init.ones(b, s, d)
    expected_spec = PartitionSpec("dp", None, None)

    # Both forward invocations, their checkpoint recomputation, and future
    # checkpoint restores must use the one lexical ``block`` scope allocated
    # during initialization.
    assert all(name.rsplit("/", 1)[0].endswith("block") for name in params)
    assert {spec for spec in layout.explain(params).values()} >= {
        PartitionSpec(None, "tp"),
        PartitionSpec("tp", None),
    }

    def native_loss(native_params, raw_values):
        result = apply_fn(native_params, Tensor(raw_values, b, s, d))
        return result.d.sum().s.sum().b.sum().unwrap()

    with mesh.jax_mesh:
        native_forward = jax.jit(
            lambda native_params, raw_values: apply_fn(native_params, Tensor(raw_values, b, s, d)).unwrap(),
            in_shardings=(layout.parameter_shardings(params), layout.tensor_sharding(values)),
            out_shardings=layout.tensor_sharding(init.ones(b, s, d[None])),
        )
        forward_and_grad = jax.jit(
            jax.value_and_grad(native_loss),
            in_shardings=(layout.parameter_shardings(params), layout.tensor_sharding(values)),
            out_shardings=(None, layout.parameter_shardings(params)),
        )
        forward = native_forward(params, values.unwrap())
        loss, grads = forward_and_grad(params, values.unwrap())

    assert jnp.isfinite(loss)
    assert all(value.sharding.spec == layout.parameter_specs(params)[name] for name, value in grads.items())
    actual = apply_fn(params, values)
    assert forward.sharding.spec == expected_spec
    assert actual.topology[0].placement == mesh.dp and actual.topology[-1].replicated

    path = tmp_path / "remat_pallas"
    ax.save(model, str(path))
    restored = ax.model(stacked_blocks, mesh=mesh)
    ax.load(str(path), target=restored)
    assert jnp.allclose(restored(values).unwrap(), actual.unwrap())
    assert restored.layout.explain(restored.params) == layout.explain(params)


def test_optimizer_slots_follow_parameter_paths_when_shapes_are_ambiguous():
    mesh = _mesh()
    model, _, _, _ = _projection_model(mesh)
    params, _, layout = ax.to_jax(model, sharding=True)

    # The model deliberately has equally shaped column- and row-parallel
    # weights.  A shape-only optimizer-state heuristic would replicate both
    # slots; Optax preserves parameter names in its slot trees, so use them.
    host_state = jax.device_get(optax.adam(1e-3).init(params))
    state_shardings = layout.state_shardings(host_state, params, strict=True)
    parameter_shardings = layout.parameter_shardings(params)

    checked = 0
    for path, sharding in jax.tree_util.tree_flatten_with_path(state_shardings)[0]:
        parameter_names = [getattr(entry, "key", None) for entry in path]
        matching = next((name for name in reversed(parameter_names) if name in params), None)
        if matching is not None:
            assert sharding == parameter_shardings[matching]
            checked += 1

    assert checked == 2 * len(params)  # Adam's first and second moments.


def test_remat_inside_repeat_preserves_tied_scope_and_parallel_gradients():
    mesh = _mesh()
    b, s, d, h = ax.b[mesh.dp](4), ax.s(4), ax.d(4), ax.h[mesh.tp](4)

    @ax.remat
    def block(values):
        return values.d.proj(h).h.proj(d[None])

    model = ax.model(lambda values: values.repeat(block, times=3), mesh=mesh).init(b, s, d)
    params, apply_fn, layout = ax.to_jax(model, sharding=True)
    values = init.ones(b, s, d)

    # ``repeat`` deliberately owns its parameters, so its tying scope must
    # dominate the nested remat scope on initialization and recomputation.
    assert all(name.startswith("repeat_block_") for name in params)

    def loss(native_params, raw_values):
        output = apply_fn(native_params, Tensor(raw_values, b, s, d))
        return output.d.sum().s.sum().b.sum().unwrap()

    with mesh.jax_mesh:
        compiled_grad = jax.jit(
            jax.grad(loss),
            in_shardings=(layout.parameter_shardings(params), layout.tensor_sharding(values)),
            out_shardings=layout.parameter_shardings(params),
        )
        grads = compiled_grad(params, values.unwrap())

    assert set(grads) == set(params)
    assert all(value.sharding.spec == layout.parameter_specs(params)[name] for name, value in grads.items())


def test_independent_remat_blocks_with_the_same_short_name_do_not_tie_explicit_parameters():
    mesh = _mesh()
    b, d = ax.b[mesh.dp](4), ax.d[mesh.tp](4)

    def make_left_block():
        @ax.remat
        def block(values):
            return values * init.ones(d).param(name="scale")

        return block

    def make_right_block():
        @ax.remat
        def block(values):
            return values * init.ones(d).param(name="scale")

        return block

    left, right = make_left_block(), make_right_block()
    model = ax.model(lambda values: left(values) + right(values), mesh=mesh).init(b, d)

    assert len(model.params) == 2
    assert any("make_left_block" in name for name in model.params)
    assert any("make_right_block" in name for name in model.params)
    assert jnp.allclose(model(init.ones(b, d)).unwrap(), 2.0)


def test_mesh_jit_caches_each_explicit_tensor_layout_independently():
    mesh = _mesh()
    placed = init.ones(ax.b[mesh.dp](8), ax.d[mesh.tp](4))
    replicated = init.ones(ax.b[mesh.dp](8), ax.d[None](4))

    @ax.jit(mesh=mesh)
    def identity(values):
        return values

    first = identity(placed)
    second = identity(replicated)
    third = identity(placed)

    assert first.unwrap().sharding.spec == PartitionSpec("dp", "tp")
    assert second.unwrap().sharding.spec == PartitionSpec("dp", None)
    assert third.unwrap().sharding.spec == PartitionSpec("dp", "tp")


def test_mesh_jit_preserves_static_argument_semantics_when_called_by_keyword():
    mesh = _mesh()
    b = ax.b[mesh.dp](8)
    x = init.ones(b)

    @ax.jit(mesh=mesh, static_argnames="double")
    def scale(values, double=False):
        return values * (2.0 if double else 1.0)

    actual = scale(x, double=True)
    assert jnp.allclose(actual.unwrap(), 2.0)
    assert actual.unwrap().sharding.spec == PartitionSpec("dp",)


def test_static_axis_arguments_compile_separate_output_layouts():
    mesh = _mesh()
    b, d = ax.b[mesh.dp](4), ax.d(4)
    values = init.ones(b, d)

    @ax.jit(mesh=mesh, static_argnames="output_axis")
    def annotate(values, output_axis):
        return Tensor(values.unwrap(), b, output_axis)

    tp_output = annotate(values, ax.h[mesh.tp](4))
    replicated_output = annotate(values, ax.h[None](4))

    assert tp_output.unwrap().sharding.spec == PartitionSpec("dp", "tp")
    assert replicated_output.unwrap().sharding.spec == PartitionSpec("dp", None)
    assert tp_output.topology[-1].placement == mesh.tp
    assert replicated_output.topology[-1].replicated


def test_export_rejects_partial_parameter_trees_without_mutating_them():
    mesh = _mesh()
    model, b, d, _ = _projection_model(mesh)
    params, apply_fn, _ = ax.to_jax(model, sharding=True)
    partial_params = dict(params)
    missing_name = next(iter(partial_params))
    del partial_params[missing_name]

    with pytest.raises(RuntimeError, match="missing from an initialized model/exported parameter dictionary"):
        apply_fn(partial_params, init.ones(b, d))

    assert missing_name not in partial_params
    assert set(partial_params) == set(params) - {missing_name}


def test_export_also_rejects_missing_direct_tensor_parameters():
    b, d = ax.b(2), ax.d(3)

    def direct_parameter(values):
        return values + init.ones(d).param(name="offset")

    model = ax.model(direct_parameter).init(b, d)
    _, apply_fn = ax.to_jax(model)

    with pytest.raises(RuntimeError, match="missing from an initialized model/exported parameter dictionary"):
        apply_fn({}, init.ones(b, d))


def test_tied_transpose_rejects_explicit_replication_mismatch():
    b, d, h = ax.b(2), ax.d(4), ax.h(6)

    def incompatible_tie(values):
        hidden = values.d.proj(h, tie="@shared", bias=False)
        return hidden.h.proj(d[None], tie="@shared", bias=False)

    with pytest.raises(ValueError, match="incompatible logical layout"):
        ax.model(incompatible_tie).init(b, d)


def test_invalid_mesh_axis_extent_fails_early_and_init_is_transactional():
    mesh = _mesh()
    b, d, h = ax.b[mesh.dp](4), ax.d(4), ax.h[mesh.tp](5)
    model = ax.model(lambda values: values.d.proj(h), mesh=mesh)

    with pytest.raises(ValueError, match="cannot be evenly sharded"):
        model.init(b, d)

    assert not model.is_initialized
    assert model.params == {}
    assert model.param_layouts == {}


def test_sharded_bundle_kernel_grad_and_normal_slicing():
    mesh = _mesh()
    b, s = ax.b[mesh.dp](4), ax.s[mesh.tp](6)
    values = init.ones(b, s)

    def tile_pair(pair):
        left, right = pair
        return (left * 2.0) & (right + 3.0)

    @ax.jit(mesh=mesh)
    def bundle_kernel(inputs):
        left, right = (inputs & inputs).s(4).map(tile_pair, interpret=True)
        return left + right

    actual = bundle_kernel(values)
    assert actual.unwrap().sharding.spec == PartitionSpec("dp", "tp")
    assert jnp.allclose(actual.unwrap(), 6.0)

    identity_model = ax.model(lambda tensor: tensor, mesh=mesh).init(b, s)
    _, _, layout = ax.to_jax(identity_model, sharding=True)

    def kernel_loss(raw_values):
        squared = Tensor(raw_values, b, s).s(4).map(lambda tile: tile * tile, interpret=True)
        return squared.s.sum().b.sum().unwrap()

    with mesh.jax_mesh:
        sharded_grad = jax.jit(
            jax.grad(kernel_loss),
            in_shardings=layout.tensor_sharding(values),
            out_shardings=layout.tensor_sharding(values),
        )(values.unwrap())

    assert sharded_grad.sharding.spec == PartitionSpec("dp", "tp")
    assert jnp.allclose(sharded_grad, 2.0)

    @ax.jit(mesh=mesh)
    def double(inputs):
        return inputs * 2.0

    assert double(values.s[mesh.tp][:4]).unwrap().sharding.spec == PartitionSpec("dp", "tp")
    with pytest.raises(ValueError, match="cannot be evenly sharded"):
        double(values.s[mesh.tp][:5])


def test_checkpoint_round_trip_restores_layout_and_reshards(tmp_path):
    mesh = _mesh()
    model, b, d, _ = _projection_model(mesh)
    expected = model(init.ones(b, d)).unwrap()
    path = tmp_path / "placed_model"
    ax.save(model, str(path))

    restored = ax.model(model.fn, mesh=mesh)
    ax.load(str(path), target=restored)
    actual = restored(init.ones(b, d)).unwrap()

    assert jnp.allclose(actual, expected)
    assert restored.layout.explain(restored.params) == model.layout.explain(model.params)
    assert all(isinstance(value.sharding, NamedSharding) for value in restored.params.values())


def test_kernel_preserves_placement_metadata_and_cpu_interpret_mode():
    mesh = _mesh()
    s = ax.s[mesh.dp](14)
    x = init.ones(s)

    @ax.jit(mesh=mesh)
    def kernel_step(values):
        return values.s(4).map(lambda block: block * 2.0, interpret=True)

    actual = kernel_step(x)

    assert actual.topology == (s,)
    assert jnp.allclose(actual.unwrap(), 2.0)
    assert actual.unwrap().sharding.spec == PartitionSpec("dp",)
    assert ax.to_jax(ax.model(lambda y: y, mesh=mesh).init(s), sharding=True)[2].tensor_spec(actual) == PartitionSpec("dp",)


def test_legacy_positional_shard_argument_has_a_clear_migration_error():
    with pytest.raises(TypeError, match="has been removed"):
        ax.jit(shard=[ax.b, ax.d])
