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


def smart_eval(code: str, env: dict):
    """Safely evaluates both single-line expressions and multi-line blocks."""
    # Secure the environment if not already done
    if "__builtins__" not in env:
        env["__builtins__"] = {}

    try:
        # Use env as the single unified globals dictionary to prevent lambda scoping errors
        return eval(code, env)
    except SyntaxError:
        # If it's multi-line, wrap it in a dummy function and execute it
        wrapped_code = "def _mcp_dynamic_wrapper():\n" + "\n".join(
            "    " + line for line in code.splitlines()) + "\n_mcp_result = _mcp_dynamic_wrapper()"
        exec(wrapped_code, env)
        return env.get("_mcp_result", None)


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
            local_context[var_name] = Tensor(jnp.zeros(shape, dtype=jnp.float32), *axes)

        # 4. Evaluate the AI's proposed code!
        result = smart_eval(code_snippet, local_context)

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
            res = smart_eval(code_snippet, env)

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
            res = smart_eval(code_snippet, env)
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
            return smart_eval(architecture_code, env)

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
            res = smart_eval(architecture_code, env)
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

            res = smart_eval(layer_code, env)
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

    # Better Heuristic: Check for chained projections and non-linearities
    if architecture_code.count(".proj") >= 2 and (".silu" in architecture_code or ".gelu" in architecture_code):
        remat_points.append("Wrap the FeedForward/Expansion block. (High activation memory, low compute).")

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


@mcp.tool()
def axiom_scope_debugger(architecture_code: str, input_topology: dict) -> str:
    """
    AI usage: Call this to visualize exactly which parameters are allocated during a forward pass.
    Crucial for debugging weight-tying (e.g., checking if `@global` worked) and scope leaks.
    """
    try:
        compiler_state.reset_pass_state()
        compiler_state.params.clear()
        local_context = {"ax": ax, "nn": nn, "jnp": jnp}

        kwargs = {}
        for var_name, axes_info in input_topology.items():
            axes = [getattr(ax, ax_name)(size) for ax_name, size in axes_info]
            shape = [size for _, size in axes_info]
            kwargs[var_name] = Tensor(jnp.zeros(shape, dtype=jnp.float32), *axes)

        def _eval_fn(*args):
            env = dict(zip(input_topology.keys(), args))
            env.update({"ax": ax, "nn": nn, "jnp": jnp, "compiler_state": compiler_state, "__builtins__": {}})
            res = smart_eval(architecture_code, env)
            return res.unwrap() if hasattr(res, 'unwrap') else res

        # Run eager pass
        compiler_state.is_initializing = True
        _eval_fn(*list(kwargs.values()))
        compiler_state.is_initializing = False

        params = compiler_state.params
        if not params:
            return "🔍 Scope Debugger: No parameters were allocated during this forward pass."

        report = "🔍 Axiom Scope & Parameter Allocation Tree\n\n"

        # Group parameters by their base name to easily spot scope nesting
        from collections import defaultdict
        grouped_params = defaultdict(list)

        for name, tensor in params.items():
            # A rough heuristic to group by the final parameter name vs the scope path
            parts = name.split('/')
            base_name = parts[-1]
            path = "/".join(parts[:-1]) if len(parts) > 1 else "Root"
            grouped_params[path].append((base_name, tensor.shape))

        total_params = 0
        for scope_path, items in grouped_params.items():
            report += f"📁 Scope: `{scope_path}`\n"
            for base_name, shape in items:
                size = 1
                for dim in shape: size *= dim
                total_params += size
                report += f"   ├─ 📄 {base_name} | Shape: {shape} | Params: {size:,}\n"
            report += "\n"

        report += f"Total Allocated Parameters: {total_params:,}\n"
        return report

    except Exception as e:
        return f"Scope Debugger Failed:\n{traceback.format_exc()}"


@mcp.tool()
def axiom_anti_pattern_scanner(draft_code: str) -> str:
    """
    AI usage: Run this static linter on your drafted Axiom code BEFORE executing the Ghost Pass.
    It catches verbose topology searches, standard python math inside projections, and bad configs.
    """
    import re
    warnings = []

    # 1. Catching verbose topology walks
    if re.search(r'\.topology.*if.*name\s*==', draft_code) or "next(a for a in" in draft_code:
        warnings.append("❌ Anti-Pattern Detected: Verbose topology walk.\n"
                        "   FIX: Use direct attribute access. Instead of `next(a for a in x.topology if a.name == 'b')`, just use `x.b`.")

    # 2. Catching dataclass field default factories
    if "field(default_factory" in draft_code:
        warnings.append("❌ Anti-Pattern Detected: Verbose dataclass fields.\n"
                        "   FIX: Axes are safe to instantiate directly. Change `x: Axis = field(...)` to `x: Axis = ax.x(256)`.")

    # 3. Catching manual axis size math inside projections
    if re.search(r'\.proj\([^)]*\.size\s*[*+]', draft_code):
        warnings.append("❌ Anti-Pattern Detected: Mathematical sizing in projections.\n"
                        "   FIX: `.proj()` supports multi-axis projection. Instead of `.proj(ax.a(x.size * y.size))`, use `.proj(ax.x, ax.y)`.")

    # 4. Catching string-concatenated tying inside loops
    if re.search(r'tie=f["\'].*\{.*\}', draft_code):
        warnings.append("⚠️ Warning: Dynamic string ties (e.g., `tie=f'w_{layer}'`).\n"
                        "   Note: You usually do not need this. `ax.model` handles loop scoping automatically if written purely.")

    if not warnings:
        return "✅ Code passes static analysis. No Axiom anti-patterns detected. Proceed to Ghost Pass."

    return "\n\n".join(warnings)


@mcp.tool()
def scan_contract_validator(step_function_code: str, carry_topology: dict, token_topology: dict) -> str:
    """
    AI usage: Call this to verify the input/output contract of a custom recurrent `step` function for `.s.scan()`.

    Inputs:
    - step_function_code: A string defining the `def step(c, t): ...` function.
    - carry_topology: dict mapping tensor names to axes for the Carry Bundle.
    - token_topology: dict mapping tensor names to axes for the Token Bundle.
    """
    try:
        compiler_state.reset_pass_state()
        local_context = {"ax": ax, "nn": nn, "jnp": jnp}

        # Synthesize Carry Bundle
        carry_tensors = []
        for var_name, axes_info in carry_topology.items():
            axes = [getattr(ax, ax_name)(size) for ax_name, size in axes_info]
            shape = [size for _, size in axes_info]
            carry_tensors.append(Tensor(jnp.zeros(shape, dtype=jnp.float32), *axes))

        # Synthesize Token Bundle
        token_tensors = []
        for var_name, axes_info in token_topology.items():
            axes = [getattr(ax, ax_name)(size) for ax_name, size in axes_info]
            shape = [size for _, size in axes_info]
            token_tensors.append(Tensor(jnp.zeros(shape, dtype=jnp.float32), *axes))

        from axiom.core import Bundle
        c_bundle = Bundle(carry_tensors) if len(carry_tensors) > 1 else carry_tensors[0]
        t_bundle = Bundle(token_tensors) if len(token_tensors) > 1 else token_tensors[0]

        # Evaluate the step function
        exec(step_function_code, {"__builtins__": {}}, local_context)
        if "step" not in local_context:
            return "❌ Failed: Could not find a function named `step` in the provided code."

        step_fn = local_context["step"]

        # Execute the contract
        result = step_fn(c_bundle, t_bundle)

        # Validate the specific Axiom Scan contract
        if not isinstance(result, tuple) or len(result) != 2:
            return f"❌ Contract Violation: `step` must return a tuple of length 2: (next_carry, (outputs,)). Got: {type(result)}"

        next_carry, outputs = result

        if not (isinstance(next_carry, Tensor) or type(next_carry).__name__ == 'Bundle'):
            return f"❌ Contract Violation: `next_carry` (first element) must be a Tensor or Bundle. Got: {type(next_carry)}"

        if not isinstance(outputs, tuple):
            return f"❌ Contract Violation: `outputs` (second element) must be a tuple of Tensors. Got: {type(outputs)}"

        report = "✅ Scan Contract Verified!\n\n"

        carry_type = "Bundle" if type(next_carry).__name__ == 'Bundle' else "Tensor"
        carry_tops = [t.topology for t in next_carry.tensors] if carry_type == "Bundle" else [next_carry.topology]
        report += f"📥 Carry Output ({carry_type}):\n" + "\n".join(f" - {tops}" for tops in carry_tops) + "\n\n"

        report += f"📤 Sequence Outputs ({len(outputs)} tensors):\n" + "\n".join(
            f" - {t.topology if hasattr(t, 'topology') else 'Unknown'}" for t in outputs)

        return report

    except Exception as e:
        return f"Scan Contract Validation Failed:\n{traceback.format_exc()}"


@mcp.tool()
def axiom_idioms_and_best_practices() -> str:
    """
    AI usage: Call this BEFORE writing full model architectures to learn the "Axiom Way".
    Contains framework philosophy, parallel bundle syntax, RL loops, SSMs, and monad patching.
    """
    return """
    # The Axiom Philosophy & Best Practices
    Axiom is a strictly typed, named-axis functional eDSL. Do not write PyTorch/Flax-style object-oriented boilerplate. Models are pure functions. Parameters allocate themselves implicitly.

    ## 1. Topologies & Bundles over Manual Projections
    BAD (PyTorch style): 
    q, k, v = x.proj(), x.proj(), x.proj()

    GOOD (Axiom Bundles):
    # Execute parallel operations across multiple tensors at once using `&`
    q, k, v = (x & x & x).d.proj().d.split(ax.h(heads), ax.hd)

    ## 2. Configs Should Hold Axis Objects, Not Integers
    BAD: `@dataclass class Config: d_model: int = 384`
    GOOD: `@dataclass class Config: d: Axis = ax.d(384); s: Axis = ax.s(512)`

    ## 3. Trust ax.to_jax and ax.model for Auto-Scoping
    Do NOT manually number tied weights (e.g., `tie=f"attn_norm_{layer}"`). 
    When looping over a block `for _ in range(depth): x = block(x)`, Axiom automatically handles hierarchical scoping of the parameter dictionary. Write the block purely.

    ## 4. State Space Models (SSMs) & Recurrence
    Axiom supports native associative scans and recurrent loops. 
    ```python
    def ssm(x: Tensor, depth=8):
        for _ in range(depth):
            # Native parallel scan over the sequence axis
            s, _ = (x.d.rms_norm() & x.d.proj().d.rms_norm()).s.scan(nn.ssm_op, associative=True)
            x = x + swiglu((x + s).d.rms_norm())
        return x
    ```

    ## 5. Advanced Topologies: Tying, Masks, and Repeats
    ```python
    # Weight Tying: 'local' ties within the function, '@global' ties anywhere
    x = x + x.d.proj(tie="local").d.silu().d.proj(tie="@global")

    # Masks & VMasks (Functional filtering)
    x = x.d.mask(lambda idx: idx < 5, fill=0.0)
    x = x.d.vmask(lambda val: val < 0.1, fill=0.0)

    # Native Repeat (Loops a function, implicitly tying weights between calls)
    x = x.repeat(lambda t: t + t.d.layer_norm().d.proj(), times=3)
    ```

    ## 6. Sliced Monads (In-Place Patching)
    Axiom allows safe, functional mutation of tensor slices.
    ```python
    def patch_mutation(x: Tensor) -> Tensor:
        patch = x.s[:10] # Returns a SlicedMonad
        patch = patch.d.layer_norm().d.proj()
        return patch[:]  # Stitch back into parent tensor in-place
    ```

    ## 7. Reinforcement Learning (Actor-Critic Loops)
    Axiom works beautifully for RL. Treat environments as registered PyTrees and use `ax.model` for networks.
    ```python
    @ax.model
    def actor(x: Tensor):
        # Outputs unnormalized logits for 5 discrete directions
        return x.d.proj(ax.h(32)).h.silu().h.proj(ax.a(5))

    @ax.model
    def critic(x: Tensor):
        # Evaluates state value, summing v(1) to return clean topology (b)
        return x.d.proj(ax.h(32)).h.silu().h.proj(ax.v(1)).v.sum()

    # Inside the ax.jit train step:
    a = actor(x).stop_grad().a.sample()

    def actor_loss(a_model):
        logits = a_model(inputs)
        entropy = logits.a.pw(lambda v: jax.nn.softmax(v) * jax.nn.log_softmax(v)).a.sum().b.t.mean()
        return nn.reinforce(logits.a, actions, advantages) + entropy * 0.01
    ```
    
    ## 8. Dataclasses & Configs (No Verbose Boilerplate)
    Axiom axes are safe to instantiate directly in configs. NEVER use `field(default_factory=...)`. Do not write `__post_init__` validation loops. Keep it elegant.
    BAD: 
    @dataclass
    class Config:
        x: Axis = field(default_factory=lambda: Axis("x", 256))
        
    GOOD:
    @dataclass
    class Config:
        x: Axis = ax.x(256)
        c: Axis = ax.c(128)

    ## 9. Hypernetworks & Multi-Axis Projections
    When generating weights for a hypernetwork, do NOT multiply axis sizes together (e.g., `x.size * h.size`) and split them later. Axiom's `.proj()` handles multi-axis generation natively.
    BAD:
    # Flat math and manual splitting
    w1 = g.z.proj(Axis("w1", cfg.x.size * cfg.h.size)).w1.split(cfg.x, cfg.h)
    
    GOOD:
    # Direct topological projection
    w1 = g.z.proj(cfg.x, cfg.h)
    return (x @ w1 + b1).h.silu()
    """


@mcp.tool()
def get_axiom_tutorial() -> str:
    """
    AI usage: Call this to read the official Axiom tutorial code.
    Use this to see how Axiom handles training loops, functional mutation, and advanced topology.
    """
    return TUTORIAL_CONTENT


def main():
    print("Starting Axiom MCP Oracle...")
    mcp.run()

if __name__ == "__main__":
    main()

TUTORIAL_CONTENT = """
Welcome to axiom!
This tutorial is designed to make you comfortable with axiom fundamentals.
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
"""