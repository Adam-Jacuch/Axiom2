import jax
import jax.numpy as jnp
from .core import TargetedTensor, Tensor, Tie
from . import init

def axiom_nn_op(func):
    """Decorator to mark a function as an Axiom native NN module."""
    func._is_axiom_nn = True
    return func

# ==========================================
# 1. STATEFUL NORMALIZATION LAYERS
# ==========================================

@axiom_nn_op
def layer_norm(targeted: TargetedTensor, tie: str = None, eps: float = 1e-5) -> Tensor:
    x = targeted.tensor
    mean = targeted.mean()

    diff = x - mean
    sq_diff = diff.square() # <-- Notice the native dynamic call!

    var = TargetedTensor(sq_diff, targeted.target_axes).mean()
    x_norm = diff / (var + eps).sqrt()

    tie_obj = Tie(tie) if tie else None
    gamma = init.ones(*targeted.target_axes).param(name="gamma", tie=tie_obj)
    beta = init.zeros(*targeted.target_axes).param(name="beta", tie=tie_obj)

    return (x_norm * gamma) + beta


@axiom_nn_op
def rms_norm(targeted: TargetedTensor, tie: str = None, eps: float = 1e-5) -> Tensor:
    x = targeted.tensor
    squared = targeted.square()
    var = TargetedTensor(squared, targeted.target_axes).mean()
    x_norm = x / (var + eps).sqrt()

    tie_obj = Tie(tie) if tie else None
    gamma = init.ones(*targeted.target_axes).param(name="gamma", tie=tie_obj)

    return x_norm * gamma


# ==========================================
# 2. LOSS FUNCTIONS (Pure Axiom Math)
# ==========================================

def _apply_reduction(loss: Tensor, reduction: str) -> Tensor:
    """Internal helper to collapse loss topologies."""
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction type: '{reduction}'. Expected 'mean', 'sum', or 'none'.")

def mse_loss(preds: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    loss = (preds - targets).square()
    return _apply_reduction(loss, reduction)


def l1_loss(preds: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    loss = (preds - targets).abs()
    return _apply_reduction(loss, reduction)


def huber_loss(preds: Tensor, targets: Tensor, delta: float = 1.0, reduction: str = 'mean') -> Tensor:
    diff = (preds - targets).abs()
    loss = diff.pw(lambda x: jnp.where(x < delta, 0.5 * jnp.square(x), delta * (x - 0.5 * delta)))
    return _apply_reduction(loss, reduction)


def bce_with_logits(logits: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    max_val = logits.clip(0, None)
    log_weight = logits.abs().pw(lambda x: jnp.log(1 + jnp.exp(-x)))
    loss = max_val - (logits * targets) + log_weight
    return _apply_reduction(loss, reduction)

def cross_entropy_loss(logits: 'TargetedTensor', targets: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Cross entropy over a targeted class axis.

    Sparse usage:
        logits:  Tensor with class axis, e.g. (s, v)
        call:    nn.cross_entropy_loss(logits.v, targets)
        targets: integer Tensor without class axis, e.g. (s)
        returns: Tensor without class axis, e.g. (s)

    Dense usage:
        logits:  Tensor with class axis, e.g. (s, v)
        call:    nn.cross_entropy_loss(logits.v, targets)
        targets: one-hot/probability Tensor with class axis, e.g. (s, v)
        returns: Tensor without class axis, e.g. (s)
    """
    import jax
    import jax.numpy as jnp
    from .core import TargetedTensor, Tensor

    if not isinstance(logits, TargetedTensor):
        raise ValueError(
            "cross_entropy_loss expects targeted logits, e.g. "
            "nn.cross_entropy_loss(out.v, targets), not nn.cross_entropy_loss(out, targets)."
        )

    if len(logits.target_axes) != 1:
        raise ValueError(
            "cross_entropy_loss expects exactly one targeted class axis, "
            f"got {[a.name for a in logits.target_axes]}."
        )

    x = logits.tensor
    class_ax = logits.target_axes[0]

    if class_ax not in x.topology:
        raise ValueError(
            f"Class axis '{class_ax.name}' is not present in logits topology "
            f"{[a.name for a in x.topology]}."
        )

    class_idx = x.topology.index(class_ax)
    log_probs_raw = jax.nn.log_softmax(x.unwrap(), axis=class_idx)

    # Topology after removing the class axis.
    out_topology = tuple(a for a in x.topology if a != class_ax)

    # Dense / one-hot case: targets include the class axis.
    if class_ax in targets.topology:
        target_raw = targets._align_to(x.topology)
        loss_raw = -(target_raw * log_probs_raw).sum(axis=class_idx)
        return Tensor(loss_raw, *out_topology)

    # Sparse integer-label case: targets do NOT include the class axis.
    target_raw = targets._align_to(out_topology).astype(jnp.int32)

    # Insert singleton class dimension so take_along_axis can gather along class_idx.
    gather_idx = jnp.expand_dims(target_raw, axis=class_idx)

    loss_raw = -jnp.take_along_axis(log_probs_raw, gather_idx, axis=class_idx)
    loss_raw = jnp.squeeze(loss_raw, axis=class_idx)

    return _apply_reduction(Tensor(loss_raw, *out_topology), reduction)


def reinforce(logits: 'TargetedTensor', actions: Tensor, advantages: Tensor, reduction: str = 'mean') -> Tensor:
    # Notice we must force 'none' on the internal cross entropy call
    # so we can multiply by the advantages before reducing!
    neg_log_prob = cross_entropy_loss(logits, actions, reduction='none')

    import jax
    safe_advantages = advantages.pw(jax.lax.stop_gradient)
    loss = neg_log_prob * safe_advantages

    return _apply_reduction(loss, reduction)


# ==========================================
# 3. ACTIVATIONS (JAX Aliases)
# ==========================================
# Exposing these allows users to natively chain: x.d.pw(nn.silu)

# Modern LLM Activations
gelu = jax.nn.gelu
silu = jax.nn.silu
swish = jax.nn.swish
mish = jax.nn.mish

# Standard Activations
relu = jax.nn.relu
leaky_relu = jax.nn.leaky_relu
sigmoid = jax.nn.sigmoid
tanh = jax.numpy.tanh
softplus = jax.nn.softplus
log_sigmoid = jax.nn.log_sigmoid

# General pointwise math
exp = jnp.exp
clip = jnp.clip
clamp = jnp.clip

# Axis-aware Reductions
# (Our .pw() wrapper will automatically inject the 'axis=' argument for these!)
softmax = jax.nn.softmax
log_softmax = jax.nn.log_softmax


def ssm_op(left: tuple, right: tuple):
    """
    Binary operator for parallel linear recurrence.
    Usage: (X & A).s.scan(nn.ssm_op, associative=True)
    Computes: h_t = A_t * h_{t-1} + X_t
    """
    X_i, A_i = left
    X_j, A_j = right

    # Axiom natively handles the broadcasting and topologies!
    return (A_j * X_i + X_j, A_j * A_i)