"""A pure-Axiom, Megatron-style decoder with manual FlashAttention-2.

Run with ``uv run python examples/megatron_flash_transformer.py``.  On two or
more devices the example selects a dp x tp mesh (two-way tensor parallelism by
default); on one device it remains executable with tp=1.  Set ``AXIOM_TP`` to
choose another tensor-parallel degree that divides both the device count and
the number of heads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from axiom import Tensor, ax, init, nn


@dataclass(frozen=True)
class Config:
    vocab_size: int = 128
    sequence_length: int = 32
    batch_per_dp: int = 2
    model_dim: int = 64
    heads: int = 4
    head_dim: int = 16
    mlp_dim: int = 128
    layers: int = 2
    block_q: int = 8
    block_k: int = 8
    steps: int = 3


def make_mesh(config: Config):
    """Pick a valid DP x TP mesh without hiding device-count assumptions."""
    devices = jax.device_count()
    requested_tp = int(os.environ.get("AXIOM_TP", "2"))
    if requested_tp < 1:
        raise ValueError("AXIOM_TP must be a positive integer.")
    tp = requested_tp if devices % requested_tp == 0 and config.heads % requested_tp == 0 else 1
    return ax.mesh(dp=devices // tp, tp=tp)


def causal_flash_attention(query: Tensor, key: Tensor, value: Tensor, *, s, sk, hd, config: Config) -> Tensor:
    """FlashAttention-2's online-softmax recurrence in Pallas SRAM.

    ``.map`` owns independent query blocks; ``.fold`` advances key/value
    blocks while keeping ``(running_max, normalizer, accumulator)`` in
    registers.  The causal predicate uses Axiom's named ``.mask()`` API.
    """

    def flash_query_tile(qkv_tile):
        q_tile, key_full, value_full = qkv_tile
        q_tile = q_tile.astype(jnp.float32)
        row_zeros = q_tile.hd.sum() * 0.0
        initial = (row_zeros - jnp.inf, row_zeros, q_tile * 0.0)

        def flash_key_block(carry, kv_tile):
            running_max, normalizer, accumulator = carry
            k_tile, v_tile = kv_tile
            scores = (q_tile.hd @ k_tile.astype(jnp.float32)) / jnp.sqrt(hd.size)
            scores = scores.s.sk.mask(
                lambda q_offset, k_offset: (
                    ax.grid[sk] * config.block_k + k_offset
                    > ax.grid[s] * config.block_q + q_offset
                ),
                fill=-jnp.inf,
            )
            next_max = running_max.maximum(scores.sk.max())
            previous_scale = (running_max - next_max).exp()
            weights = (scores - next_max).exp()
            return (
                next_max,
                previous_scale * normalizer + weights.sk.sum(),
                previous_scale * accumulator + (weights.sk @ v_tile.astype(jnp.float32)),
            )

        _, normalizer, accumulator = (key_full & value_full).sk(config.block_k).fold(
            flash_key_block,
            init=initial,
            stages=2,
        )
        return accumulator / normalizer

    # Only sequence is a Pallas grid axis: b[dp] and h[tp] are inherited from
    # the surrounding GSPMD shard, while each program owns one query block.
    return (query & key & value).s(config.block_q).map(flash_query_tile)


def make_decoder(config: Config, mesh):
    dp, tp = mesh.dp, mesh.tp
    b = ax.b[dp](config.batch_per_dp * mesh.axis_sizes["dp"])
    s, sk = ax.s(config.sequence_length), ax.sk(config.sequence_length)
    d = ax.d(config.model_dim)
    h, hd = ax.h[tp](config.heads), ax.hd(config.head_dim)
    ff = ax.ff[tp](config.mlp_dim)
    vocab = ax.vocab(config.vocab_size)

    if config.model_dim != config.heads * config.head_dim:
        raise ValueError("model_dim must equal heads * head_dim.")
    if config.sequence_length % config.block_q or config.sequence_length % config.block_k:
        raise ValueError("This compact example uses sequence lengths divisible by both FlashAttention tile sizes.")

    def attention(x: Tensor) -> Tensor:
        # Column-parallel fused QKV: the head axis is placed on tp.
        qkv = x.d.proj(ax.qkv(3), h, hd)
        query, key, value = qkv.qkv
        key, value = (key & value).s.rename(sk)
        context = causal_flash_attention(query, key, value, s=s, sk=sk, hd=hd, config=config)

        # Row-parallel output projection: h x hd becomes a tp-sharded d,
        # then the projection returns the replicated residual d axis.
        context = context.h.hd.merge(ax.d[tp](config.model_dim))
        return context.d[tp].proj(d[None])

    def swiglu(x: Tensor) -> Tensor:
        # Column-parallel expansion and row-parallel contraction.
        up, gate = x.d.proj(ax.gate(2), ff).gate
        return (up * gate.ff.silu()).ff.proj(d[None])

    @ax.remat
    def block(x: Tensor) -> Tensor:
        x = x + attention(x.d.rms_norm())
        return x + swiglu(x.d.rms_norm())

    def decoder(tokens: Tensor) -> Tensor:
        positions = init.normal(s, d).param(name="position_embedding")
        x = nn.embed(tokens, vocab, d) + positions
        for _ in range(config.layers):
            x = block(x)
        return x.d.rms_norm().d.proj(vocab)

    return decoder, (b, s, d, vocab)


def run(config: Config = Config()):
    mesh = make_mesh(config)
    decoder, (b, s, d, vocab) = make_decoder(config, mesh)
    model = ax.model(decoder, mesh=mesh).init(b, s)
    params, apply_fn, layout = ax.to_jax(model, sharding=True)
    optimizer = optax.adamw(learning_rate=3e-4, weight_decay=0.01)
    opt_state = layout.place_state(optimizer.init(params), params)

    token_key, target_key = jax.random.split(jax.random.key(0))
    tokens = Tensor(jax.random.randint(token_key, (b.size, s.size), 0, vocab.size), b, s)
    targets = Tensor(jax.random.randint(target_key, (b.size, s.size), 0, vocab.size), b, s)
    tokens = Tensor(jax.device_put(tokens.unwrap(), layout.input_sharding(tokens)), b, s)
    targets = Tensor(jax.device_put(targets.unwrap(), layout.input_sharding(targets)), b, s)

    param_shardings = layout.parameter_shardings(params)
    state_shardings = layout.state_shardings(opt_state, params)

    def train_step(params, opt_state, tokens, targets):
        def loss_fn(current_params):
            logits = apply_fn(current_params, tokens)
            return nn.cross_entropy_loss(logits.vocab, targets).unwrap()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    with mesh.jax_mesh:
        train_step = jax.jit(
            train_step,
            in_shardings=(param_shardings, state_shardings, layout.input_sharding(tokens), layout.input_sharding(targets)),
            out_shardings=(param_shardings, state_shardings, None),
        )

    print(f"mesh={mesh}; batch={b.size}; heads={config.heads}")
    for step in range(config.steps):
        params, opt_state, loss = train_step(params, opt_state, tokens, targets)
        print(f"step={step:02d} loss={float(loss):.4f}")


if __name__ == "__main__":
    run()
