from axiom import ax, nn, Tensor, init, axiom_jit, axiom_step
import jax.numpy as jnp
import optax

def attention(x: Tensor):
    """attention: standard transformer attention"""
    q, k, v = (x & x & x).d.proj(bias=False)
    q, k = (q & k).d.rms_norm()
    k, v = (k & v).s.rename(ax.sk)
    scores = (q @ k / jnp.sqrt(x.d.size)).sk.s.mask(lambda sk, s: sk > s, fill=-1e9).sk.softmax()
    return scores @ v

def swiglu(x: Tensor):
    """swiglu: swish gated linear unit"""
    u, l = (x & x).d.proj()
    return (u.silu() * l).d.proj()

def block(x: Tensor):
    """block: standard transformer block"""
    return x + swiglu(attention(x))

@axiom_jit
def transformer(x: Tensor, depth: int = 8):
    """transformer"""
    for _ in range(depth):
        x = block(x)
    return x

optim = optax.adam(1e-3)

@axiom_step(model=transformer, optimizer=optim)
def step(x, y):
    """step: standard transformer step"""
    out = transformer(x)
    loss = nn.bce_with_logits(out, y).b.s.d.mean()
    transformer.vjp(loss)
    return loss

def data():
    """data loader"""
    for _ in range(100):
        x = init.normal(ax.b(4), ax.s(16), ax.d(32))
        y = init.normal(ax.b(4), ax.s(16), ax.d(32))
        yield x.d.softmax(), y.d.softmax()

for x, y in data():
    loss = step(x, y)
    print(f"loss: {loss}") # dataset is purely random, so loss will not decrease here!