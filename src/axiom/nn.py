import jax
import jax.numpy as jnp
from .core import TargetedTensor, Tensor, ax, Tie
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
    """Standard Layer Normalization with Learnable Gamma and Beta."""
    x = targeted.tensor
    mean = targeted.mean()

    # 1. Use pure Axiom native broadcasting for the difference!
    diff = x - mean
    sq_diff = diff.pw(jnp.square)

    # 2. Re-target the squared difference to reduce over the correct axes programmatically
    var = TargetedTensor(sq_diff, targeted.target_axes).mean()

    # 3. Normalize
    x_norm = diff / (var + eps).pw(jnp.sqrt)

    tie_obj = Tie(tie) if tie else None
    gamma = init.ones(*targeted.target_axes).param(name="gamma", tie=tie_obj)
    beta = init.zeros(*targeted.target_axes).param(name="beta", tie=tie_obj)

    return (x_norm * gamma) + beta


@axiom_nn_op
def rms_norm(targeted: TargetedTensor, tie: str = None, eps: float = 1e-5) -> Tensor:
    """Root Mean Square Normalization (Used in Llama, Hawk, Mamba)."""
    x = targeted.tensor

    # 1. Square the tensor (Returns a pure Tensor)
    squared = targeted.pw(jnp.square)

    # 2. Re-target the squared tensor programmatically so it knows what to reduce!
    var = TargetedTensor(squared, targeted.target_axes).mean()

    # 3. Pure tensors natively support .pw(), so this works perfectly
    x_norm = x / (var + eps).pw(jnp.sqrt)

    tie_obj = Tie(tie) if tie else None
    gamma = init.ones(*targeted.target_axes).param(name="gamma", tie=tie_obj)

    return x_norm * gamma


# ==========================================
# 2. LOSS FUNCTIONS (Pure Axiom Math)
# ==========================================

def mse_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """Mean Squared Error. Automatically broadcasts topologies if needed."""
    return (preds - targets).pw(jnp.square)


def bce_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """Numerically stable Binary Cross Entropy with Logits."""
    # max(x, 0) - x * z + log(1 + exp(-abs(x)))
    max_val = logits.pw(lambda x: jnp.clip(x, 0, None))
    log_weight = logits.pw(jnp.abs).pw(lambda x: jnp.log(1 + jnp.exp(-x)))
    return max_val - (logits * targets) + log_weight


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

# Axis-aware Reductions
# (Our .pw() wrapper will automatically inject the 'axis=' argument for these!)
softmax = jax.nn.softmax
log_softmax = jax.nn.log_softmax