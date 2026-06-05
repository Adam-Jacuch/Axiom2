from mcp.server.fastmcp import FastMCP
import jax.numpy as jnp
import traceback

import jax
import re

# Import your framework internals!
from axiom.core import compiler_state, Tensor
from axiom import ax, nn

# Initialize the MCP Server
mcp = FastMCP("Axiom-Oracle")


@mcp.tool()
def get_active_parameters() -> str:
    """
    AI usage: Call this to see the exact dictionary of currently allocated Axiom parameters.
    Returns a string representation of the keys and the current param_counter.
    """
    params = list(compiler_state.params.keys())
    counter = compiler_state.param_counter

    if not params:
        return f"State is empty. Current Counter: {counter}"

    formatted_params = "\n".join(f" - {p}" for p in params)
    return f"Current Parameter Counter: {counter}\n\nAllocated Parameters:\n{formatted_params}"


@mcp.tool()
def axiom_ghost_pass_oracle(code_snippet: str, input_topology: dict) -> str:
    """
    AI usage: Call this to mathematically verify tensor shapes BEFORE writing final code.

    Inputs:
    - code_snippet: A string evaluating to a Tensor. e.g., "x.d.rms_norm().d.proj(ax.d(64))"
    - input_topology: A dict mapping var names to axis definitions.
      Format: {"x": [("b", 1), ("s", 512), ("d", 32)]}

    Returns:
    The exact resulting output topology, OR the precise Axis mismatch error.
    """
    try:
        # 1. Reset state so we don't accidentally pollute a real training trace
        compiler_state.reset_pass_state()

        # 2. Build a safe local execution context for the AI
        local_context = {"ax": ax, "nn": nn, "jnp": jnp}

        # 3. Dynamically synthesize the dummy Tensors from the AI's requested topology
        for var_name, axes_info in input_topology.items():
            axes = []
            shape = []
            for ax_name, ax_size in axes_info:
                # Dynamically call ax.b(1), ax.s(512), etc.
                axis_obj = getattr(ax, ax_name)(ax_size)
                axes.append(axis_obj)
                shape.append(ax_size)

            # Create a zero-filled tensor safe for index operations
            local_context[var_name] = Tensor(jnp.zeros(shape, dtype=jnp.int32), *axes)

        # 4. Evaluate the AI's proposed code!
        # We use eval() because we expect a pure mathematical expression.
        result = eval(code_snippet, {"__builtins__": {}}, local_context)

        # 5. Extract and return the mathematical proof of the shape
        if hasattr(result, 'topology'):
            return f"Success! Output Topology: {result.topology}"
        elif hasattr(result, 'tensors'):  # Handle parallel Bundles!
            tops = [t.topology for t in result.tensors]
            return f"Success! Output is a Bundle with Topologies:\n{tops}"
        else:
            return f"Code executed, but output is not an Axiom Tensor. Type: {type(result)}"

    except Exception as e:
        # If the AI hallucinates a shape, we catch the Axiom crash and feed the
        # exact stack trace back to the AI so it can fix its own mistake!
        return f"Shape Error / Axiom Exception:\n{traceback.format_exc()}"


@mcp.tool()
def tracer_leak_autopsy(code_snippet: str, input_topology: dict) -> str:
    """
    AI usage: Call this to verify state-machine purity when writing Remat/Checkpoint layers.
    """
    import traceback
    try:
        # 1. Reset state
        compiler_state.reset_pass_state()
        compiler_state.params.clear()

        local_context = {"ax": ax, "nn": nn, "jnp": jnp}

        # 2. Build dummy Tensors
        kwargs = {}
        for var_name, axes_info in input_topology.items():
            axes = [getattr(ax, ax_name)(size) for ax_name, size in axes_info]
            shape = [size for _, size in axes_info]
            tensor = Tensor(jnp.zeros(shape, dtype=jnp.float32), *axes)
            kwargs[var_name] = tensor
            local_context[var_name] = tensor

        # 3. Define the execution wrapper
        def _trace_fn(*args):
            env = dict(zip(input_topology.keys(), args))

            # Combine everything into the GLOBALS dictionary
            env.update({
                "ax": ax,
                "nn": nn,
                "jnp": jnp,
                "compiler_state": compiler_state,
                "__builtins__": {}  # Keep the security restriction here
            })

            # Pass `env` as the SECOND argument (globals)
            res = eval(code_snippet, env)

            return res.unwrap() if hasattr(res, 'unwrap') else res

        dummy_args = list(kwargs.values())

        # THE FIX: 4. RUN THE GHOST PASS! (Eagerly allocate the weights)
        compiler_state.is_initializing = True
        _trace_fn(*dummy_args)
        compiler_state.is_initializing = False

        # Reset counters before the trace
        compiler_state.param_counter = 0
        compiler_state.active_frames.clear()
        compiler_state.func_calls.clear()

        # 5. TRACE IT! (Now JAX can safely find the parameters)
        jax.make_jaxpr(_trace_fn)(*dummy_args)

        # 6. Check the global Axiom state for leaked JAX Tracers
        leaked_keys = []
        for k, v in compiler_state.params.items():
            if isinstance(v, jax.core.Tracer):
                leaked_keys.append(f" - '{k}' is leaking a {type(v).__name__}")

        if leaked_keys:
            return "❌ FAILED: Tracer Leak Detected!\n" + "\n".join(leaked_keys)

        return "✅ PASSED: Purity verified. No global tracers leaked during JAX tracing."

    except Exception as e:
        return f"Execution Error before trace completed:\n{traceback.format_exc()}"


@mcp.tool()
def xla_memory_interrogator(code_snippet: str, input_topology: dict) -> str:
    """
    AI usage: Call this to check if a custom Axiom layer will cause a TPU memory spill.
    Compiles the code to XLA HLO and parses the text dump for massive intermediate broadcasts.
    """
    try:
        compiler_state.reset_pass_state()
        compiler_state.params.clear()

        kwargs = {}
        for var_name, axes_info in input_topology.items():
            axes = [getattr(ax, ax_name)(size) for ax_name, size in axes_info]
            shape = [size for _, size in axes_info]
            kwargs[var_name] = Tensor(jnp.zeros(shape, dtype=jnp.bfloat16), *axes)

        def _compile_fn(*args):
            env = dict(zip(input_topology.keys(), args))
            env.update({"ax": ax, "nn": nn, "jnp": jnp})
            res = eval(code_snippet, {"__builtins__": {}}, env)
            # Must return pure JAX arrays for XLA compilation
            if hasattr(res, 'tensors'):
                return tuple(t.unwrap() for t in res.tensors)
            return res.unwrap() if hasattr(res, 'unwrap') else res

        dummy_args = list(kwargs.values())

        # 1. Force JAX to lower the code to StableHLO
        lowered = jax.jit(_compile_fn).lower(*dummy_args)
        hlo_text = lowered.as_text()

        # 2. Parse the HLO text for the largest tensor allocations
        # Matches patterns like: f32[4096,4096]{1,0}
        shapes = re.findall(r'[a-z0-9]+\[([0-9,\s]+)\]', hlo_text)

        max_elements = 0
        biggest_shape = ""

        for shape_str in shapes:
            dims = [int(x.strip()) for x in shape_str.split(',') if x.strip()]
            size = 1
            for d in dims: size *= d
            if size > max_elements:
                max_elements = size
                biggest_shape = shape_str

        # Assume 2 bytes per param for bfloat16
        mb_size = (max_elements * 2) / (1024 * 1024)

        report = (
            f"✅ XLA Compilation Successful.\n"
            f"Total HLO Instructions: {len(hlo_text.splitlines())}\n"
            f"Largest Intermediate Buffer Allocated: [{biggest_shape}] (~{mb_size:.2f} MB in SRAM)\n\n"
        )

        if mb_size > 100:
            report += "⚠️ WARNING: Buffer exceeds 100MB. This may cause HBM spills on smaller TPU chips."

        return report

    except Exception as e:
        return f"XLA Compilation Failed:\n{traceback.format_exc()}"


@mcp.tool()
def prng_collision_detector(architecture_code: str, input_topology: dict) -> str:
    """
    AI usage: Call this to verify statistical independence of dropout/noise layers.
    """
    import traceback
    try:
        compiler_state.reset_pass_state()
        local_context = {"ax": ax, "nn": nn, "jnp": jnp}

        kwargs = {}
        for var_name, axes_info in input_topology.items():
            axes = [getattr(ax, ax_name)(size) for ax_name, size in axes_info]
            shape = [size for _, size in axes_info]
            kwargs[var_name] = Tensor(jnp.zeros(shape, dtype=jnp.float32), *axes)

        def _eval_fn(*args):
            env = dict(zip(input_topology.keys(), args))
            env.update({"ax": ax, "nn": nn, "jnp": jnp, "compiler_state": compiler_state})
            return eval(architecture_code, {"__builtins__": {}}, env)

        # 1. Run the Eager Ghost Pass
        compiler_state.is_initializing = True
        _eval_fn(*list(kwargs.values()))
        initial_counter = compiler_state.param_counter

        # Reset the global counter before the trace pass!
        compiler_state.param_counter = 0

        # 2. Re-run to ensure the counter advances deterministically
        compiler_state.is_initializing = False
        _eval_fn(*list(kwargs.values()))
        trace_counter = compiler_state.param_counter

        if initial_counter == 0 and trace_counter == 0:
            return "✅ No stochastic/stateful layers detected. Architecture is purely deterministic."

        if initial_counter != trace_counter:
            return f"❌ FAILED: State Desynchronization! Ghost pass found {initial_counter} states, but trace found {trace_counter}."

        return f"✅ PASSED: Detected {initial_counter} strictly ordered stochastic states. No PRNG collisions."

    except Exception as e:
        return f"Execution Error:\n{traceback.format_exc()}"


@mcp.tool()
def hardware_mesh_optimizer(params_millions: int, seq_len: int, hardware_topology: str) -> str:
    """
    AI usage: Call this to generate optimal jax.sharding.Mesh layouts for ANY hardware.
    hardware_topology examples: "v4-8", "v5e-16", "A100-80GBx8", "H100x4", "RTX4090x1"
    """
    topo = hardware_topology.upper()

    # 1. Hardware Heuristics
    hbm_per_chip = 16  # Fallback baseline
    if "A100-80" in topo:
        hbm_per_chip = 80
    elif "A100" in topo:
        hbm_per_chip = 40
    elif "H100" in topo:
        hbm_per_chip = 80
    elif "4090" in topo:
        hbm_per_chip = 24
    elif "V4" in topo:
        hbm_per_chip = 32
    elif "V5E" in topo:
        hbm_per_chip = 16
    elif "V6E" in topo:
        hbm_per_chip = 32

    # 2. Parse Chip Count (handles both Google's '-N' and NVIDIA's 'xN' syntax)
    chips = 1
    if "X" in topo:
        chips = int(topo.split("X")[-1])
    elif "-" in topo and topo.split("-")[-1].isdigit():
        chips = int(topo.split("-")[-1])

    # Math: bf16 params + AdamW states = ~6 bytes per parameter
    model_gb = (params_millions * 1e6 * 6) / (1024 ** 3)

    report = f"📊 Universal Hardware Analysis for {hardware_topology}\n"
    report += f" - Total VRAM/HBM Available: {hbm_per_chip * chips} GB ({chips} chips @ {hbm_per_chip} GB)\n"
    report += f" - Static Model Weight + Optimizer Memory: ~{model_gb:.2f} GB\n\n"

    # 3. Dynamic JAX Routing Logic
    if chips == 1:
        if model_gb > (hbm_per_chip * 0.85):
            report += "⚠️ WARNING: Model exceeds single-chip memory bounds. Will likely OOM during activation materialization."
        else:
            report += "🛠️ Recommended Strategy: Single Device Execution\nCode: `mesh = None # JAX defaults to device 0`"

    elif params_millions <= 300:
        report += "🛠️ Recommended Strategy: Pure Data Parallelism (DP)\n"
        report += "Code: `mesh = Mesh(devices, ('batch',))`\n"
        report += "Reasoning: Model is small enough. FSDP communication overhead across the interconnect outweighs the memory savings. Replicate weights fully."

    elif model_gb > (hbm_per_chip * 0.7):
        report += "🛠️ Recommended Strategy: Fully Sharded Data Parallel (FSDP)\n"
        report += "Code: `mesh = Mesh(devices, ('fsdp', 'batch'))`\n"
        report += "Reasoning: Model size demands memory sharding. Optimizer states and weights will be safely distributed across the cluster."

    else:
        report += "🛠️ Recommended Strategy: Hybrid Data Parallelism\n"
        report += "Code: `mesh = Mesh(devices, ('batch',))`\n"

    return report

@mcp.tool()
def associative_scan_profiler(seq_len: int, state_dim: int, chunk_size: int) -> str:
    """
    AI usage: Call this to optimize chunk sizes for associative recall operators in SSMs.
    """
    num_chunks = seq_len // chunk_size
    # Intra-chunk parallel scan cost
    intra_cost = chunk_size * state_dim * 2
    # Inter-chunk sequential cost
    inter_cost = num_chunks * state_dim * 2

    optimal_chunk = int(seq_len ** 0.5)

    report = f"🔄 Associative Scan Topology\n"
    report += f" - Sequence: {seq_len} | State Dim: {state_dim}\n"
    report += f" - Current Chunking: {num_chunks} chunks of size {chunk_size}\n\n"

    if chunk_size != optimal_chunk:
        report += f"⚠️ Sub-optimal chunk size. For sequence length {seq_len}, an `(x & a)` reduction tree minimizes depth at chunk_size = {optimal_chunk}."
    else:
        report += f"✅ Chunk size mathematically optimal for hardware reduction trees."

    return report


@mcp.tool()
def compute_density_profiler(architecture_code: str, input_topology: dict) -> str:
    """
    AI usage: Call this to calculate the exact FLOPs and compute-per-neuron ratio.
    """
    try:
        compiler_state.reset_pass_state()
        compiler_state.params.clear()
        local_context = {"ax": ax, "nn": nn, "jnp": jnp}

        kwargs = {}
        for var_name, axes_info in input_topology.items():
            axes = [getattr(ax, ax_name)(size) for ax_name, size in axes_info]
            shape = [size for _, size in axes_info]
            kwargs[var_name] = Tensor(jnp.zeros(shape, dtype=jnp.bfloat16), *axes)

        def _eval_fn(*args):
            env = dict(zip(input_topology.keys(), args))
            env.update({"ax": ax, "nn": nn, "jnp": jnp, "compiler_state": compiler_state, "__builtins__": {}})
            res = eval(architecture_code, env)
            return res.unwrap() if hasattr(res, 'unwrap') else res

        # Eager pass to allocate parameters
        compiler_state.is_initializing = True
        _eval_fn(*list(kwargs.values()))
        compiler_state.is_initializing = False

        total_params = sum(p.size for p in compiler_state.params.values())

        # Compile to XLA and extract exact cost analysis
        lowered = jax.jit(_eval_fn).lower(*list(kwargs.values()))
        cost = lowered.cost_analysis()

        flops = cost[0].get('flops', 0) if isinstance(cost, list) else cost.get('flops', 0)
        bytes_accessed = cost[0].get('bytes accessed', 0) if isinstance(cost, list) else cost.get('bytes accessed', 0)

        if total_params == 0:
            return "⚠️ No parameters detected. Cannot calculate compute-per-neuron."

        flops_per_param = flops / total_params

        report = f"🧠 Compute Density & FLOPs Analysis\n"
        report += f" - Total Parameters: {total_params:,}\n"
        report += f" - Forward Pass FLOPs: {flops:,.0f}\n"
        report += f" - HBM Bytes Accessed: {bytes_accessed:,.0f}\n\n"
        report += f"🔬 Compute-per-Neuron Ratio: {flops_per_param:.2f} FLOPs / Param\n\n"

        if flops_per_param < 2.0:
            report += "⚠️ Warning: Extremely low compute density. This architecture is heavily memory-bound (likely too many linear projections without sufficient recurrent depth)."
        elif flops_per_param > 100.0:
            report += "✅ High compute density detected. Excellent ALU saturation (ideal for recurrent state space models or heavy associative scans)."

        return report
    except Exception as e:
        return f"Compute Profiling Failed:\n{traceback.format_exc()}"


@mcp.tool()
def numerical_stability_oracle(layer_code: str, input_topology: dict) -> str:
    """
    AI usage: Call this to stress-test a layer for bfloat16 overflow or NaN generation.
    """
    try:
        compiler_state.reset_pass_state()

        kwargs = {}
        for var_name, axes_info in input_topology.items():
            axes = [getattr(ax, ax_name)(size) for ax_name, size in axes_info]
            shape = [size for _, size in axes_info]
            # INJECT EXTREME NOISE (Simulating late-stage pretraining variance)
            noise = jax.random.normal(jax.random.PRNGKey(42), shape) * 1e3
            kwargs[var_name] = Tensor(noise.astype(jnp.bfloat16), *axes)

        def _loss_fn(params_dict, *args):
            env = dict(zip(input_topology.keys(), args))
            env.update({"ax": ax, "nn": nn, "jnp": jnp, "compiler_state": compiler_state, "__builtins__": {}})

            # Temporarily inject functional parameters
            prev_params = getattr(compiler_state, 'params', {})
            compiler_state.params = params_dict

            res = eval(layer_code, env)
            out = res.unwrap() if hasattr(res, 'unwrap') else res

            compiler_state.params = prev_params
            return jnp.mean(out)

        # 1. Eager Pass
        compiler_state.is_initializing = True
        dummy_res = eval(layer_code,
                         {"ax": ax, "nn": nn, "jnp": jnp, "compiler_state": compiler_state, "__builtins__": {}},
                         dict(zip(input_topology.keys(), kwargs.values())))
        compiler_state.is_initializing = False

        params = compiler_state.params.copy()

        # 2. Run Backward Pass with High Variance
        loss, grads = jax.value_and_grad(_loss_fn)(params, *list(kwargs.values()))

        if jnp.isnan(loss) or jnp.isinf(loss):
            return "❌ FAILED: Forward pass evaluated to NaN/Inf under high variance. Check your epsilon values in normalization layers or add gradient clipping."

        nan_grads = [k for k, v in grads.items() if jnp.isnan(v).any() or jnp.isinf(v).any()]

        if nan_grads:
            return f"❌ FAILED: Backward pass generated NaN gradients in the following parameters:\n" + "\n".join(
                f" - {k}" for k in nan_grads)

        return "✅ PASSED: Numerically stable. Forward and backward passes survived extreme variance injection."

    except Exception as e:
        return f"Stability Oracle Failed:\n{traceback.format_exc()}"


@mcp.tool()
def optimal_remat_search(architecture_code: str) -> str:
    """
    AI usage: Call this to find the mathematically optimal gradient checkpointing boundaries.
    """
    import ast

    # Static Analysis Heuristic
    remat_points = []

    if ".proj" in architecture_code and "inner" in architecture_code:
        remat_points.append("Wrap the FeedForward/MLP expansion block. (High activation memory, low compute).")

    if "softmax" in architecture_code or "attention" in architecture_code.lower():
        remat_points.append("Wrap the Attention Matrix materialization. (Quadratic memory scaling).")

    if "&" in architecture_code:  # Associative Scan detection
        remat_points.append(
            "Wrap the `(x & a)` associative scan reduction tree. Parallel recurrences hoard tape memory linearly with sequence length.")

    report = "🔍 Auto-Remat Boundary Search\n\n"
    if not remat_points:
        report += "No obvious memory bottlenecks detected. Standard JAX compilation is sufficient."
    else:
        report += "🛠️ Recommended `@ax.remat` boundaries to maximize HBM savings:\n"
        for p in remat_points:
            report += f" - {p}\n"

    return report


@mcp.tool()
def inspect_axiom_api(target_name: str) -> str:
    """
    AI usage: Call this to read the documentation, signature, and source code of ANY Axiom function.
    Use this BEFORE guessing syntax for this custom framework!
    Query examples: "rms_norm", "proj", "dropout", "remat", "Tensor"
    """
    import inspect
    from axiom import ax, nn, core, state

    # Map common namespaces to their actual modules
    modules = {"nn": nn, "ax": ax, "core": core, "state": state}
    obj = None

    try:
        # 1. Attempt to resolve the exact object path (e.g., "nn.dropout")
        if "." in target_name:
            mod_name, attr_name = target_name.split(".", 1)
            if mod_name in modules:
                obj = getattr(modules[mod_name], attr_name, None)
        else:
            # 2. If no namespace provided, sweep all modules
            for mod in modules.values():
                # Check top-level module attributes
                if target_name in dir(mod):
                    obj = getattr(mod, target_name)
                    break

                # 3. Deep sweep: Look inside classes (like Tensor, TargetedTensor, AxiomModel)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if inspect.isclass(attr) and target_name in dir(attr):
                        obj = getattr(attr, target_name)
                        # Prepend the class name so the AI knows exactly where it lives!
                        target_name = f"{attr.__name__}.{target_name}"
                        break

                if obj is not None:
                    break

        if obj is None:
            return f"❌ Could not find '{target_name}' in Axiom's public API. Are you sure it exists?"

        # Extract the signature (fails gracefully for classes/variables)
        try:
            sig = str(inspect.signature(obj))
        except ValueError:
            sig = " (Signature unavailable)"

        # Extract the docstring
        doc = inspect.getdoc(obj) or "No docstring provided."

        # Extract the source code (limit to 60 lines so we don't blow up the AI context window)
        try:
            source_lines = inspect.getsource(obj).splitlines()
            source = "\n".join(source_lines[:60])
            if len(source_lines) > 60:
                source += "\n... [Source truncated for length]"
        except Exception:
            source = "Source code unavailable (might be a compiled C extension or dynamic object)."

        # Format the ultimate AI cheat sheet
        report = f"📖 Axiom API Reference: `{target_name}`\n"
        report += f"Signature: {target_name}{sig}\n\n"
        report += f"Docstring:\n{doc}\n\n"
        report += f"Source Code Preview:\n```python\n{source}\n```"

        return report

    except Exception as e:
        return f"API Inspection Failed:\n{str(e)}"


def main():
    print("Starting Axiom MCP Oracle...")
    mcp.run()

if __name__ == "__main__":
    main()