# axiom

**A strictly typed, named-axis eDSL built on JAX.**

Axiom rethinks tensor calculus and neural network design by completely dropping the object-oriented boilerplate of traditional frameworks. Models are just functions. Operations are routed strictly through named topological axes. Parameters allocate themselves dynamically during the JAX trace.

Kill the boilerplate. Name your axes.

---

## ⚡ Installation

Axiom requires Python 3.10+ and relies heavily on JAX and Optax. Since Axiom is currently in active development, install it directly from GitHub:

```bash
pip install git+https://github.com/Adam-Jacuch/Axiom2.git
```

**Hardware Acceleration:**
Install the version tailored to your hardware using the optional dependency flags. When installing directly from GitHub, use the following syntax:

```bash
pip install "axiom-jax[cuda13] @ git+https://github.com/Adam-Jacuch/Axiom2.git"  # For NVIDIA GPUs (CUDA 13)
pip install "axiom-jax[cuda12] @ git+https://github.com/Adam-Jacuch/Axiom2.git"  # For NVIDIA GPUs (CUDA 12)
pip install "axiom-jax[apple] @ git+https://github.com/Adam-Jacuch/Axiom2.git"   # For Apple Silicon (Metal)
pip install "axiom-jax[tpu] @ git+https://github.com/Adam-Jacuch/Axiom2.git"     # For Google TPUs
```

Alternatively, you can clone the repository and install it locally (recommended for active development):
```bash
git clone https://github.com/Adam-Jacuch/Axiom2.git
cd Axiom2
pip install -e ".[cuda13]"  # Replace cuda13 with your target hardware
```

---

## 🌐 Explicit multi-device layouts

Distributed placement is attached to the logical axes that own it. Create one
explicit mesh, annotate axes with its tokens, and keep ordinary slicing exactly
as it is. A logical axis is placed on at most one mesh axis; represent a true
2D decomposition by splitting it into two named logical axes.

```python
import jax.numpy as jnp
from axiom import ax, init, nn

mesh = ax.mesh(dp=8, tp=8)
dp, tp = mesh.dp, mesh.tp

b, s = ax.b[dp](1024), ax.s(2048)
d, h, hd = ax.d(8192), ax.h[tp](64), ax.hd(128)

def megatron_mha(x):
    # Column-parallel QKV: heads are partitioned over tp.
    qkv = x.d.proj(ax.qkv(3), h, hd)
    q, k, v = qkv.qkv
    k, v = (k & v).s.rename(ax.sk)

    scores = (q.hd @ k) / (hd.size ** 0.5)
    scores = scores.s.sk.mask(lambda query, key: key > query, fill=-jnp.inf)
    context = scores.sk.softmax().sk @ v

    # Row-parallel output projection: merge back to a tp-sharded feature
    # axis, then return a replicated residual feature axis.
    context = context.h.hd.merge(ax.d[tp](d.size))
    return x + context.d[tp].proj(ax.d[None])

model = ax.model(megatron_mha, mesh=mesh).init(b, s, d)
x = init.normal(b, s, d)

@ax.jit(mesh=mesh)
def forward(model, x):
    return model(x)

# Placement and ordinary Axiom slices compose naturally.
prefix = x.d[tp][:128]
```

`ax.jit(shard=...)` and mutable `Axis.shard(...)` are removed. They guessed a
mesh from a positional list and could not describe parameter or optimizer
layouts. The explicit form above is the single sharding API.

Use `None` to explicitly require replication while still inferring an axis
size: `x.d[tp].proj(ax.d[None])` is a row-parallel TP→replicated projection.
When that output axis is not already present, give it a size once—for example
`x.ff[tp].proj(ax.d(D)[None])`.

For native JAX/Optax pipelines, request the layout companion when exporting:

```python
params, apply_fn, layout = ax.to_jax(model, sharding=True)
params = layout.place_params(params)
opt_state = optimizer.init(params)  # zeros_like state inherits parameter placement

# For a custom optimizer state tree, use the same rules explicitly.
opt_state = layout.place_state(opt_state, params)
```

`layout.parameter_specs(params)`, `layout.parameter_shardings(params)`,
`layout.input_sharding(tensor)`, and `layout.output_sharding(tensor)` expose
the exact JAX objects used at the boundary. Axiom checkpoints save this logical
layout metadata and re-place parameters on the receiving mesh during
`ax.load(..., target=model)`.

---

## 🧠 The Core Philosophy

In Axiom, tensors are aware of their own topology. You don't perform operations on shapes; you target specific axes. 

```python
from axiom import ax, nn, init, Tensor

def mlp(x: Tensor) -> Tensor:
    # 1. Target the 'd' axis
    # 2. Project it
    # 3. Add a bias
    # 4. Apply SiLU activation
    return x.d.proj().d.bias().d.silu()
```
Notice there are no `__init__` blocks, no `self.linear = nn.Linear(...)`, and no explicit parameter shapes. Axiom uses a "Ghost Pass" during compilation to auto-infer shapes and allocate your weight dictionary cleanly in the background.

---

## 🔥 Superpowers

### 1. Parallel Bundles
Use the `&` operator to bundle tensors together and execute operations in parallel.

```python
def qkv_attention(x: Tensor) -> Tensor:
    # Project x into 3 parallel tensors, then topologically split the 'd' axis!
    q, k, v = (x & x & x).d.proj().d.split(ax.heads(8), ax.h_dim)
    
    # Contract axes safely and explicitly
    k, v = (k & v).s.rename(ax.sk)
    scores = (q @ k) / (q.h_dim.size ** 0.5)
    attn = scores.sk.softmax()
    
    # Merge the topology back into a flat dimension
    return (attn.sk @ v).heads.h_dim.merge(ax.d).d.proj()
```

### 2. Weight Tying & Native Recurrence
Need to share weights? Just give them a `tie` name. Need to run an RNN? Use `.repeat()`. Axiom will automatically lock the compiler scope to ensure the exact same parameters are used on every pass.

```python
def ssm_block(x: Tensor) -> Tensor:
    # Native parallel scan over the sequence axis using shared weights
    s, _ = (x.d.rms_norm() & x.d.proj().d.rms_norm()).s.scan(nn.ssm_op, associative=True)
    
    # Tie projections across different parts of the network
    return x + x.d.proj(tie="shared_mlp").d.silu().d.proj(tie="@global")
```

### 3. Sliced Monads (In-Place Patching)
Axiom allows you to slice a tensor, mutate that specific chunk via a `SlicedMonad`, and stitch it back into the parent tensor—all functionally and safely.

```python
def mutate_prefix(x: Tensor) -> Tensor:
    # Extract the first 10 tokens of the sequence axis
    patch = x.s[:10]
    
    # Mutate the patch
    patch = patch.d.layer_norm().d.proj()
    
    # Stitch it back into the parent tensor in-place
    return patch[:]
```

---

## ⚙️ Named-Axis Pallas Kernels

Axiom can lower tiled named-axis programs to `jax.experimental.pallas.pallas_call` without exposing positional `BlockSpec` boilerplate. Tile the axes that own the parallel output, then use `.map()` to define one Pallas program. Bundles remain bundles inside the kernel body.

For a complete DP×TP decoder training example, see [megatron_flash_transformer.py](examples/megatron_flash_transformer.py): it implements causal FlashAttention-2 directly with named `.map()`/`.fold()` and trains it through native JAX/Optax sharding trees.

```python
out = (left & right).m(128).n(128).map(
    lambda pair: pair[0] @ pair[1]     # contracts the shared k axis
)
```

Inside a map, `ax.grid[axis]` is the named program coordinate and `ax.tile[axis]` is the tile-local register index vector. `.fold()` performs a sequential, register-resident loop inside the program; the fold coordinate is also available through `ax.grid`.

```python
import jax.numpy as jnp
from axiom import ax, Tensor

def causal_flash_tile(qkv):
    q, k, v = qkv                      # q: [b, h, s, d], k/v: [b, h, sk, d]
    q = q.astype(jnp.float32)           # accumulate in FP32 registers
    zeros = q.d.sum() * 0.0
    init = (
        zeros - jnp.inf,                   # online-softmax max: [b, h, s]
        zeros,                              # online-softmax normalizer: [b, h, s]
        q * 0.0,                            # value accumulator: [b, h, s, d]
    )

    def attend_key_tile(carry, kv):
        max_so_far, normalizer, accumulator = carry
        k_tile, v_tile = kv
        scores = (q @ k_tile.astype(jnp.float32)) / jnp.sqrt(q.d.size)
        scores = scores.s.sk.mask(
            lambda q_local, k_local: (
                ax.grid[ax.sk] * 64 + k_local
                > ax.grid[ax.s] * 64 + q_local
            ),
            fill=-jnp.inf,
        )

        next_max = max_so_far.maximum(scores.sk.max())
        previous_weight = (max_so_far - next_max).exp()
        weights = (scores - next_max).exp()
        return (
            next_max,
            previous_weight * normalizer + weights.sk.sum(),
            previous_weight * accumulator + (weights @ v_tile.astype(jnp.float32)),
        )

    _, normalizer, accumulator = (k & v).sk(64).fold(
        attend_key_tile,
        init=init,
        until=ax.grid[ax.s] + 1,
        stages=2,
    )
    return accumulator / normalizer

# One parallel program per [batch, head, query-block].
out = (q & k & v).b(1).h(1).s(64).map(causal_flash_tile)
```

Tail tiles are supported: the last program receives a fixed-size block with deterministic zero padding, and out-of-bounds stores are discarded. CPU runs automatically use Pallas interpret mode; GPU/TPU runs use native lowering. `stages` is a portable pipeline hint, with backend-specific asynchronous buffering controlled by the active Pallas backend.

---

## 🚀 60-Second Training Loop

Axiom models are pure functions, but we provide lightweight wrappers to integrate seamlessly with standard JAX transformations and Optax optimizers.

```python
import optax
from axiom import ax, nn, init, Tensor

def model_fn(x: Tensor) -> Tensor:
    return x.d.proj(ax.h(128)).h.silu().h.proj(ax.d(64))

# Initialize the model wrapper and optimizer
model = ax.model(model_fn)
optim = optax.adamw(1e-3)
state = None

@ax.jit
def train_step(model, state, x, y):
    def loss_fn(m):
        preds = m(x)
        return nn.mse_loss(preds, y)

    # Automatically routes gradients through the functional PyTree
    loss, grads = ax.value_and_grad(loss_fn)(model)
    model, state = ax.apply_updates(model, grads, optim, state)
    
    return model, state, loss

# Initialize dummy data with explicit axes
x, y = init.normal(ax.xy(2), ax.b(32), ax.d(64)).xy

for step in range(100):
    model, state, loss = train_step(model, state, x, y)
```

---

## 🧠 Axiom AI Oracle (MCP Server)

Axiom includes a native **Model Context Protocol (MCP)** server. Instead of fighting with AI hallucinations, you can hook your IDE directly into the Axiom JAX compiler. 

When installing from GitHub:
```bash
pip install "axiom-jax[cuda13,ai] @ git+https://github.com/Adam-Jacuch/Axiom2.git"
```

When activated, your AI assistant (Claude, GPT, Gemini) gains the ability to:
* **Mathematically verify matrix shapes** before writing code.
* **Profile XLA TPU/GPU memory** to prevent HBM spills.
* **Detect Tracer Leaks** and verify state-machine purity via the JAX tape.
* **Calculate Compute-per-Neuron (FLOPs)** for custom architectures.

### How to Activate in Cursor / Windsurf
1. Ensure you installed Axiom with the AI flag (e.g., `pip install -e ".[ai]"`).
2. Open your IDE Settings and navigate to **MCP Servers**.
3. Add a new server:
   * **Name:** `Axiom Oracle`
   * **Type:** `command`
   * **Command:** `axiom-oracle` (or `uv run axiom-oracle`)

*That's it. Ask your AI to "Write an associative scan block and check the compute density," and watch it talk to the framework natively.*

---

## 🤝 Pure JAX Interop

Don't want to be locked into an ecosystem? Axiom models can be instantly exported into pure JAX `(params, apply_fn)` paradigms at any time.

```python
# Convert the Axiom eDSL model into standard JAX dictionaries and functions
jax_params, apply_fn = ax.to_jax(model, *input_axises)

# Now you can use standard jax.vjp, jax.grad, or flax/haiku tools!
```
