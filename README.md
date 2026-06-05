# axiom

**A strictly typed, named-axis eDSL built on JAX.**

Axiom rethinks tensor calculus and neural network design by completely dropping the object-oriented boilerplate of traditional frameworks. Models are just functions. Operations are routed strictly through named topological axes. Parameters allocate themselves dynamically during the JAX trace.

Kill the boilerplate. Name your axes.

---

## ⚡ Installation

Axiom requires Python 3.10+ and relies heavily on JAX and Optax. Since Axiom is currently in active development, install it directly from GitHub:

```bash
pip install git+[https://github.com/Adam-Jacuch/Axiom2.git](https://github.com/Adam-Jacuch/Axiom2.git)
```

**Hardware Acceleration:**
Install the version tailored to your hardware using the optional dependency flags. When installing directly from GitHub, use the following syntax:

```bash
pip install "axiom-jax[cuda13] @ git+[https://github.com/Adam-Jacuch/Axiom2.git](https://github.com/Adam-Jacuch/Axiom2.git)"  # For NVIDIA GPUs (CUDA 13)
pip install "axiom-jax[cuda12] @ git+[https://github.com/Adam-Jacuch/Axiom2.git](https://github.com/Adam-Jacuch/Axiom2.git)"  # For NVIDIA GPUs (CUDA 12)
pip install "axiom-jax[apple] @ git+[https://github.com/Adam-Jacuch/Axiom2.git](https://github.com/Adam-Jacuch/Axiom2.git)"   # For Apple Silicon (Metal)
pip install "axiom-jax[tpu] @ git+[https://github.com/Adam-Jacuch/Axiom2.git](https://github.com/Adam-Jacuch/Axiom2.git)"     # For Google TPUs
```

Alternatively, you can clone the repository and install it locally (recommended for active development):
```bash
git clone [https://github.com/Adam-Jacuch/Axiom2.git](https://github.com/Adam-Jacuch/Axiom2.git)
cd Axiom2
pip install -e ".[cuda13]"  # Replace cuda13 with your target hardware
```

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
pip install "axiom-jax[cuda13,ai] @ git+[https://github.com/Adam-Jacuch/Axiom2.git](https://github.com/Adam-Jacuch/Axiom2.git)"
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