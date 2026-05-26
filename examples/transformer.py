from axiom import ax, nn, init, Tensor
import jax
import jax.numpy as jnp

def rope(x: Tensor, base=10000.0):
    """rope: rotary positional embedding"""
    hd = ax.hd(x.h.size // 2)
    x1, x2 = x.h.split(ax.sp(2), hd).sp
    rotated = ax.stack([-x2, x1], ax.sp(2)).sp.hd.merge(ax.h(x.h.size))
    t = init.arange(x.s.size, ax.s)
    powers = init.arange(0, x.h.size, 2, hd) / x.h.size
    inv_freq = 1.0 / (base ** powers)
    freqs = t * inv_freq
    freqs_full = ax.stack([freqs, freqs], ax.sp(2)).sp.hd.merge(ax.h(x.h.size))
    cos, sin = freqs_full.pw(jnp.cos), freqs_full.pw(jnp.sin)
    return (x * cos) + (rotated * sin)

def gqa(x: Tensor, kv_heads=2, q_heads=8):
    """gqa: grouped query attention"""
    assert q_heads % kv_heads == 0
    head_dim = x.d.size // q_heads
    k, v = (x & x).d.proj(ax.kvh(kv_heads), ax.h(head_dim))
    q = x.d.proj(ax.qh(q_heads // kv_heads), ax.kvh(kv_heads), ax.h(head_dim))
    k, q = (k & q).h.rms_norm()
    k, q = rope(k), rope(q)
    k, v = (k & v).s.rename(ax.sk)
    scores = (q @ k / jnp.sqrt(k.h.size)).s.sk.mask(lambda s, sk: sk > s, fill=-1e9).sk.softmax()
    return (scores.sk @ v).kvh.qh.h.proj(ax.d(x.d))

def swiglu(x: Tensor):
    """swiglu: swish gated linear unit"""
    u, l = (x & x).d.proj()
    return (u.d.silu() * l).d.proj()

def transformer(x: Tensor, depth=8):
    """transformer: modern transformer"""
    for _ in range(depth):
        x = x + gqa(x.d.rms_norm())
        x = x + swiglu(x.d.rms_norm())
    return x

params, model = ax.to_jax(transformer, ax.b(1), ax.s(32), ax.d(16))

@jax.jit # thanks to ax.to_jax, we can treat our model *like* native JAX
def step(params, x, y, lr=1e-2):
    def loss_fn(params):
        logits = model(params, x)
        return nn.mse_loss(logits, y).unwrap()

    loss, fn = jax.vjp(loss_fn, params)
    grads, = fn(lr)

    params = jax.tree.map(lambda p, g: p - g, params, grads)

    return params, loss

x, y = init.normal(ax.xy(2), ax.b(4), ax.s(32), ax.d(16)).xy

for i in range(100):
    params, loss = step(params, x, y)
    print(f"{i}: {loss:.4f}")

