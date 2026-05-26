from axiom import ax, nn, init, Tensor
import optax

def swiglu(x: Tensor):
    """swiglu: swish gated linear unit"""
    u, l = (x & x).d.proj()
    return (u.d.silu() * l).d.proj()

def ssm(x: Tensor, depth=8):
    """ssm: state space model"""
    for _ in range(depth):
        s, _ = (x.d.rms_norm() & x.d.proj().d.rms_norm()).s.scan(nn.ssm_op, associative=True)
        x = x + swiglu((x + s).d.rms_norm())
    return x

model = ax.model(ssm)
optim = optax.adamw(1e-3)
state = None

@ax.jit
def step(model, state, x, y):
    def loss_fn(model):
        logits = model(x)
        return nn.mse_loss(logits, y)

    loss, grads = ax.value_and_grad(loss_fn)(model)
    model, state = ax.apply_updates(model, grads, optim, state)

    return model, state, loss

x, y = init.normal(ax.xy(2), ax.b(4), ax.s(32), ax.d(16)).xy

for i in range(100):
    model, state, loss = step(model, state, x, y)
    print(f"{i}: {loss:.4f}")