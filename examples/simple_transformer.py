from axiom import ax, nn, Tensor
import jax, optax

def mha(x: Tensor, heads=8):
    """mha: multi head attention (prefer gqa for causal inference)"""
    q, k, v = (x & x & x).d.proj().d.split(ax.h(heads), ax.dh)
    k, q = (k & q).dh.rope(seq_ax=ax.s)
    out = nn.flash_attention(q, k, v, seq_ax=ax.s, head_ax=ax.h)
    return out.h.dh.proj(ax.d(x.d))

def swiglu(x: Tensor):
    """swiglu: swish gated linear unit"""
    u, l = (x & x).d.proj()
    return (u.d.silu() * l).d.proj()

@ax.remat # native rematerialization/checkpoinitng
def block(x: Tensor):
    """block: standard transformer block"""
    x = x + mha(x.d.rms_norm())
    return x + swiglu(x.d.rms_norm())

def transformer(x: Tensor, depth=8):
    """transformer: modern transformer"""
    x = nn.embed(x, ax.v(8), ax.d(16))
    for _ in range(depth):
        x = block(x)
    return x.d.proj(ax.v(8))

params, model = ax.to_jax(transformer, ax.b(1), ax.s(32))
optim = optax.adamw(1e-3)
state = optim.init(params)

@jax.jit
def step(params, state, x, y):
    def loss_fn(params):
        logits = model(params, x)
        return nn.cross_entropy_loss(logits.v, y).unwrap()

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, state = optim.update(grads, state, params)
    params = optax.apply_updates(params, updates)

    return params, state, loss

x = Tensor([[1, 4, 2, 6, 3, 2, 6, 5, 7, 7, 0, 3]], ax.b(1), ax.s(12))
y = Tensor([[6, 2, 5, 3, 5, 4, 7, 6, 1, 0, 5, 1]], ax.b(1), ax.s(12))

for i in range(100):
    params, state, loss = step(params, state, x, y)
    print(f"{i}: {loss:.4f}")
