from axiom import ax, nn, Tensor, init
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

def transformer(x: Tensor, depth: int = 8):
    """transformer"""
    for _ in range(depth):
        x = block(x)
    return x

model = ax.model(transformer)
optim = optax.adam(1e-3)
state = None

@ax.jit
def step(model, state, x, y):
    """step: standard transformer step"""
    def loss_fn(model):
        out = model(x)
        return nn.bce_with_logits(out, y).b.s.d.mean()
    loss, grads = ax.value_and_grad(loss_fn)(model)
    model, state = ax.apply_updates(model, grads, optim, state)
    return model, state, loss

def data():
    """data loader"""
    x = init.normal(ax.b(4), ax.s(16), ax.d(32))
    y = init.normal(ax.b(4), ax.s(16), ax.d(32))
    for _ in range(100):
        yield x.d.softmax(), y.d.softmax()

for x, y in data():
    model, state, loss = step(model, state, x, y)
    print(f"loss: {loss:.4f}") # dataset is purely random, so loss will not decrease here!