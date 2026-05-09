# ⩘ Axiom 

**A mathematically pure, topology-driven deep learning compiler.**

Axiom completely eliminates the most common sources of bugs in modern AI research—brittle tensor indices and manual shape broadcasting—by treating tensors not as blind arrays of numbers, but as **topologically aware mathematical concepts**.

Built natively on top of JAX, Axiom allows you to write imperative, modular code that AOT-compiles into pure, stateless XLA execution graphs. It is designed from the ground up to support the next generation of sub-quadratic architectures, state-space models, and massive TPU parallelization.

---

## 🏛️ The Four Pillars of Axiom

### 1. Topological Routing
Stop tracking arbitrary integers. Axiom replaces integer-based dimension indexing with named topological axes. Operations natively target the mathematical dimension, regardless of where it physically lives in memory. Whether you are flattening spatial dimensions into patches or splitting embeddings into multi-head attention, you simply name the axes you wish to manipulate.

### 2. Native Broadcasting
The manual dimension alignment nightmare is over. Axiom features a custom dynamic union-engine that automatically aligns batch and feature dimensions mathematically. If you multiply a batch of rewards against a batch-sequence log-probability, Axiom identifies the common batch axis and broadcasts the rewards across the sequence natively.

### 3. AOT Functional JIT
Define highly complex, stateful architectures dynamically. Axiom traces imperative weight initializations and compiles them entirely into pure, XLA-optimized functional graphs. This provides the flexibility of a research-oriented API with the zero-overhead performance of a static compiler.

### 4. The Step Monad
Scale across hardware effortlessly. The step interceptor automatically manages JAX gradient tracers dynamically. You get the clean, stateful syntax of an imperative framework with the blistering distributed speed and functional purity of JAX and Optax.

---

## ⚡ Key Features

**Native Mixed Precision**
Seamlessly hook into hardware Tensor Cores. Axiom allows you to cast activations and initialize projections natively in bfloat16 or float16 to optimize memory bandwidth and compute throughput.

**Associative Scans**
First-class support for parallelized associative scans. This unlocks the true sub-quadratic potential required for modern State-Space Models (SSMs) and Recurrent Neural Networks, executing them at the speed of XLA.

**Topological Convolutions**
No messy C++ kernels or manual stride-tricks. Axiom dynamically unwraps sliding windows across any sequence natively. By combining unfolding with topological projection, 1D and 2D convolutions become simple, high-level operations.

**Causal Masking**
Native boolean coordinate grids allow you to restrict autoregressive flow across any targeted sequence axis. Masking is treated as a first-class mathematical operation rather than a post-processing hack.

---

## 🚀 Architectural Overview

**The Project-And-Map Workflow**
In Axiom, a "Layer" is simply a projection from one set of topological axes to another. For example, a linear layer is a projection from an input dimension axis to an output dimension axis. Because Axiom handles the flattening and reconstruction of these axes automatically, N-to-M dimensional mappings are handled with a single, clear call.

**The State Interceptor**
Axiom solves the "State Problem" in functional programming by using a global compiler state that only activates during the JIT-trace. This allows weights to be defined exactly where they are used, eliminating the need for complex parameter-passing boilerplate found in traditional functional libraries.

---

## 🛠️ Installation and Requirements

Axiom requires JAX and Optax for its backend execution and optimization. It is recommended to use the UV package manager for high-performance dependency resolution.

1. Clone the repository.
2. Create a virtual environment.
3. Install Axiom in editable mode to begin developing your custom architectures.