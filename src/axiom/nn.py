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


def infonce_loss(query: Tensor, key: Tensor, batch_ax: 'Axis', temp: float = 0.07, reduction: str = 'mean') -> Tensor:
    """
    InfoNCE (Contrastive) Loss.
    Automatically computes the cross-batch similarity matrix and applies cross entropy.
    """
    from .core import Axis, TargetedTensor
    from . import init

    # 1. Rename the key's batch axis to prevent it from broadcasting linearly!
    batch_ax_k = Axis(batch_ax.name + "_k", batch_ax.size)
    key_renamed = TargetedTensor(key, (batch_ax,)).rename(batch_ax_k)

    # 2. Compute similarities
    # Axiom automatically contracts the shared embedding axes (e.g., 'd')
    # and leaves a pure (batch_ax, batch_ax_k) similarity matrix!
    logits = (query @ key_renamed) / temp

    # 3. The correct labels are the diagonal indices (0, 1, 2, ... N-1)
    targets = init.arange(batch_ax.size, batch_ax)

    # 4. Cross Entropy over the renamed key batch axis!
    # Because cross_entropy_loss already handles _apply_reduction, we just return it directly.
    return cross_entropy_loss(getattr(logits, batch_ax_k.name), targets, reduction=reduction)


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


def embed(tokens: Tensor, vocab_ax: 'Axis', embed_ax: 'Axis', tie: str = None, init=None) -> Tensor:
    """
    Lookup table for discrete tokens.
    Usage: x_emb = nn.embed(tokens, ax.vocab, ax.d)
    """
    from .state import state
    from . import init as ax_init
    from .core import Tensor

    if vocab_ax.size is None or embed_ax.size is None:
        raise ValueError("Embedding requires strict sizes for both the vocabulary and embedding axes.")

    def default_init(key, shape):
        return ax_init.normal(key, shape) * 0.02

    initializer = init if init is not None else default_init

    # 1. Allocate the continuous weight matrix (vocab_size, d_model)
    emb_raw = state.get_param("embedding", (vocab_ax.size, embed_ax.size), initializer, tie=tie)
    embedding_matrix = Tensor(emb_raw, vocab_ax, embed_ax)

    # 2. Use our beautifully restored RoutedContext to safely gather the vectors!
    return getattr(embedding_matrix, vocab_ax.name)[tokens].gather()

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


@axiom_nn_op
def rope(targeted: TargetedTensor, seq_ax: 'Axis', tie: str = None, base: float = 10000.0,
         rot_fraction: float = 1.0) -> Tensor:
    """
    Rotary Position Embedding (supports partial rotation).
    """
    from . import init
    import jax.numpy as jnp
    from .core import Axis, TargetedTensor

    x = targeted.tensor
    feat_ax = targeted.target_axes[0]

    # Dynamically resolve the true sequence axis from the tensor's topology!
    resolved_seq_ax = next((a for a in x.topology if a.name == seq_ax.name), None)
    if resolved_seq_ax is None or resolved_seq_ax.size is None:
        raise ValueError(f"Sequence axis '{seq_ax.name}' must exist and have a known size in the tensor topology.")

    rot_dim = int(feat_ax.size * rot_fraction)

    # 1. Handle Partial Rotation Splitting
    if rot_fraction < 1.0:
        x_rot = targeted[:rot_dim]
        x_pass = targeted[rot_dim:]
        rot_ax = Axis(feat_ax.name, rot_dim)
    else:
        x_rot = x
        x_pass = None
        rot_ax = feat_ax

    # 2. Slice the rotation chunk in half
    half_dim = rot_dim // 2
    x1 = TargetedTensor(x_rot, (rot_ax,))[:half_dim]
    x2 = TargetedTensor(x_rot, (rot_ax,))[half_dim:]

    # 3. Rotate (-x2, x1) and join
    rotated = getattr((-x2) & x1, rot_ax.name).join()

    # 4. Generate Frequencies (Using the safely resolved axis!)
    t = init.arange(resolved_seq_ax.size, resolved_seq_ax)
    half_ax = Axis("_half", half_dim)
    powers = init.arange(0, rot_dim, 2, half_ax) / rot_dim
    inv_freq = 1.0 / (base ** powers)
    freqs = t * inv_freq

    # Duplicate frequencies for both halves and join
    joined_freqs = getattr(freqs & freqs, half_ax.name).join()
    freqs_full = getattr(joined_freqs, half_ax.name).rename(rot_ax)

    cos, sin = freqs_full.pw(jnp.cos), freqs_full.pw(jnp.sin)

    # 5. Apply RoPE
    x_out = (x_rot * cos) + (rotated * sin)

    # 6. Re-stitch if partial
    if x_pass is not None:
        stitched = getattr(x_out & x_pass, rot_ax.name).join()
        return getattr(stitched, rot_ax.name).rename(feat_ax)

    return x_out


def flash_attention(query: 'Tensor', key: 'Tensor', value: 'Tensor',
                    seq_ax: 'Axis', head_ax: 'Axis',
                    mask: 'Tensor' = None) -> 'Tensor':
    """
    Memory-efficient Flash Attention (Auto-fused by XLA).
    Automatically contracts the hidden dimension between Q and K.
    """
    import jax.nn as jnn
    from .core import Tensor

    # 1. Topological Verification
    if seq_ax not in query.topology or head_ax not in query.topology:
        raise ValueError(f"Query must contain seq_ax '{seq_ax.name}' and head_ax '{head_ax.name}'.")

    # 2. Extract the hidden dimension dynamically (the axis that isn't batch, seq, or heads)
    q_hidden_axes = [a for a in query.topology if a not in (seq_ax, head_ax) and "batch" not in a.name.lower()]
    if not q_hidden_axes:
        raise ValueError("Could not infer the hidden dimension axis for the dot product.")

    hidden_ax = q_hidden_axes[0]

    # 3. Unwrap the raw arrays for XLA
    q_raw, k_raw, v_raw = query.unwrap(), key.unwrap(), value.unwrap()
    mask_raw = mask.unwrap() if mask is not None else None

    # 4. Execute JAX's native attention
    out_raw = jnn.dot_product_attention(
        q_raw, k_raw, v_raw,
        mask=mask_raw
    )

    # 5. Restore topological safety
    return Tensor(out_raw, *query.topology)


def dropout(x: 'Tensor', rate: float = 0.1, training: bool = True) -> 'Tensor':
    """
    Axiom-native dropout.
    Uses the deterministic compiler state to guarantee unique PRNG keys per layer!
    """
    if not training or rate == 0.0:
        return x

    import jax
    import jax.numpy as jnp
    from .state import state
    from .core import compiler_state, Tensor

    keep_rate = 1.0 - rate

    # 1. Guarantee a mathematically unique key for this specific layer
    # We fold the global layer counter into the root key!
    layer_key = jax.random.fold_in(state.root_key, compiler_state.param_counter)

    # Advance the counter so the next dropout layer gets a new key
    compiler_state.param_counter += 1

    # 2. Generate the mask matching the tensor's raw physical shape
    mask = jax.random.bernoulli(layer_key, keep_rate, shape=x.unwrap().shape)

    # 3. Apply mask and scale
    dropped_raw = jnp.where(mask, x.unwrap() / keep_rate, 0.0)

    return Tensor(dropped_raw, *x.topology)