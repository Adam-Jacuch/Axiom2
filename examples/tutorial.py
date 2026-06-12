"""
Welcome to axiom!
This tutorial is designed to make you comfortable with axiom fundamentals.
"""
import optax

from axiom import ax, nn, init, Tensor

# in axiom, models are functional and therefore represented as functions
def g(x: Tensor) -> Tensor:
    # axiom uses named axises, and supports all standard functions
    # parameters are made implicitly here
    x = x.d.proj().d.bias().d.gate().d.silu()

    # you can also explicitly define your own parameters (and register them via param())
    w = init.normal(ax.d(x.d), ax.d2(x.d * 2)).param(name="@super")
    b = init.normal(ax.d2(x.d * 2)).param()

    # explicit tensor contractions. by default, we contract over the rightmost shared axis
    x = w @ x + b
    # but we can also contract over an explicit axis(es) if desired
    # x = w.d @ x + b

    # for functions that allocate params, you can override their default inits via init=
    return x.d2.silu().d2.proj(ax.d(x.d2 // 2), init=init.normal * 0.01)

# for breakpoint debugging, you can use ax.trace to walk a dummy tensor through the model
# axiom naturally supports a robust debugger experience, walking you through how the tensor topology mutates
@ax.trace(ax.b(4), ax.s(32), ax.d(16))
def f(x: Tensor) -> Tensor:
    # function composition. new parameters are allocated for each call.
    return g(g(g(x)))
    # to weight tie, do 'return x.repeat(g, times=3)'

# advanced topology: bundles, split & merge
def self_attention(x: Tensor, heads=4) -> Tensor:
    # bundles: execute parallel operations across multiple tensors at once using `&`
    # here, we project our inputs into q, k, and v simultaneously.
    # then we topologically split the 'd' axis into heads (h) and h_dim (hd) in parallel!
    q, k, v = (x & x & x).d.proj().d.split(ax.h(heads), ax.hd)
    # to weight tie the projection of all three, use tie= (works on any function that allocates params)
    # q, k, v = (x & x & x).d.proj(tie='w').d.split(ax.heads(4), ax.h_dim)

    # this is the main 'weirdness' coming from pytorch/jax
    k, v = (k & v).s.rename(ax.sk)
    # the reason for doing this is because we want scores to have shape (s, s)
    # so we have to tell axiom that the two s's are unique, so that it does not share them after contraction
    # i.e. [b, s, d] @ [b, s, d] -> [b, s], but [b, s, d] @ [b, sk, d] -> [b, s, sk]
    # similarly, we do v.s -> v.sk so that at the end v contracts over sk, bringing it back to [b, s, d]
    # i.e. [b, s, sk, h].sk @ [b, sk, h, dh] -> [b, s, h, dh]

    # scale dot product attention using explicit axis contractions
    # contract over h_dim, then apply softmax over the sequence (s) axis
    scores = (q @ k) / (q.hd.size ** 0.5)
    # to make this causal, use mask(), which uses indexed based masking
    # scores = scores.sk.s.mask(lambda sk, s: sk > s, fill=-1e9)
    # this says 'wherever the key comes from after the query (future), mask it'
    attn = scores.sk.softmax()

    # contract attention weights with values
    # finally, merge the 'heads' and 'h_dim' axes back into a flat 'd' axis
    out = (attn.sk @ v).h.hd.merge(ax.d)
    return out.d.proj()

# weight tying, masks, & recurrence
def advanced_block(x: Tensor) -> Tensor:
    # weight tying: use the 'tie' argument to explicitly share memory
    x = x + x.d.proj(tie="local").d.silu().d.proj(tie="@global")

    # now, to re-use a 'tied' weight, simple call tie with the same name
    # ex. x = x.d.proj(tie="local") will reuse the weight earlier
    # local weights can tie within the same function scope
    # global weights (starting with '@') tie anywhere (and self tie if you call a function multiple times)

    # mask: filter tensors functionally based on their own index
    x = x.d.mask(lambda idx: idx < 5, fill=0.0)

    # vmask: filter tensors functionally based on their own values
    x = x.d.vmask(lambda val: val < 0.1, fill=0.0)

    # repeat: loop a function natively. axiom will automatically tie weights
    # ex. here norm and proj weights are tied between calls (good for rnns)
    x = x.repeat(lambda t: t + t.d.layer_norm().d.proj(), times=3)

    return x

# slices & monads: mutating patches
def patch_mutation(x: Tensor) -> Tensor:
    # taking a slice returns a "SlicedMonad" (a targeted view of the chunk)
    patch = x.s[:10]

    # perform operations purely on the chunked patch
    patch = patch.d.layer_norm().d.proj()

    # stitch the safe patch back into the parent tensor in-place
    return patch[:]
    # important: you can only patch back via [:] if you preserved the exact topology of the slice
    # otherwise, stitching back is non-trivial and you have to do that manually via join() or something
    # ex. x = x.d[-5:].proj(ax.a(2))[:] will not work

# training
# call ax.model to properly initialize the model for jax
model = ax.model(f)
optim = optax.sgd(1e-3, momentum=0.9)
state = None

# or ax.jit(shard=[ax.b, ax.d]) for FSDP, or @ax.jit(static_argnames=var) for conditions
@ax.jit # call axiom jit rather than jax jit if you use an axiom model so that implicit parameters get allocated properly
def step(model, state, x, y):
    def loss_fn(model):
        out = model(x)
        return nn.mse_loss(out, y)

    loss, grads = ax.value_and_grad(loss_fn)(model)

    fixed_grads = {}
    for name, grad in grads.items():
        if "super" in name:
            fixed_grads[name] = grad * 0.1
        else:
            fixed_grads[name] = grad

    model, state = ax.apply_updates(model, fixed_grads, optim, state)

    return model, state, loss

# natively tuple unpack a targeted axis
x, y = init.normal(ax.xy(2), ax.b(32), ax.d(64)).xy

for i in range(1000):
    model, state, loss = step(model, state, x, y)
    print(f"{i}: {loss:.4f}")