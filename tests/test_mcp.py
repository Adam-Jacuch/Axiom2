import pytest
import jax.numpy as jnp
from axiom.core import compiler_state
from axiom import ax, Tensor

# Import the raw functions directly from your MCP server!
from axiom.mcp import get_active_parameters, axiom_ghost_pass_oracle


def test_mcp_get_active_parameters():
    print("--- Testing MCP: Active Parameters ---")
    compiler_state.params.clear()
    compiler_state.reset_pass_state()

    # 1. Test Empty State
    empty_report = get_active_parameters()
    assert "State is empty" in empty_report
    assert "Current Counter: 0" in empty_report

    # 2. Inject dummy parameters
    compiler_state.params = {"layer_0/gamma_0": jnp.ones(10), "layer_0/bias_1": jnp.zeros(10)}
    compiler_state.param_counter = 2

    # 3. Test Populated State
    pop_report = get_active_parameters()
    assert "layer_0/gamma_0" in pop_report
    assert "Current Parameter Counter: 2" in pop_report


def test_mcp_ghost_pass_oracle():
    print("--- Testing MCP: Ghost Pass Oracle ---")
    compiler_state.params.clear()
    compiler_state.reset_pass_state()

    # 1. The AI proposes a valid Axiom operation
    valid_code = "(x & x).d.proj(ax.d(32))"
    topology_dict = {
        "x": [("b", 2), ("s", 16), ("dh", 4), ("d", 16)]
    }

    result = axiom_ghost_pass_oracle(valid_code, topology_dict)

    # It should successfully compute the topology without crashing
    assert "Success" in result
    assert "'b'=2" in result
    assert "'d'=32" in result

    # 2. The AI hallucinates a mathematically invalid operation
    invalid_code = "x.d.proj(ax.d(32)).fake_method_that_doesnt_exist()"

    error_result = axiom_ghost_pass_oracle(invalid_code, topology_dict)

    # It should catch the exception and return the traceback to the AI
    assert "Shape Error / Axiom Exception:" in error_result
    assert "fake_method_that_doesnt_exist" in error_result


from axiom.mcp import tracer_leak_autopsy, xla_memory_interrogator


def test_mcp_tracer_leak_autopsy():
    print("--- Testing MCP: Tracer Leak Autopsy ---")

    topology_dict = {"x": [("b", 1), ("s", 32), ("d", 64)]}

    # 1. Test a pure, mathematically safe block
    safe_code = "x.d.rms_norm()"
    safe_result = tracer_leak_autopsy(safe_code, topology_dict)
    assert "PASSED" in safe_result

    # 2. Force an intentional leak by saving a parameter dynamically without declaring it!
    # (This simulates the exact bug we fixed in the checkpointing logic earlier)
    leak_code = "(lambda t: [compiler_state.params.update({'leaky_param': t.unwrap() * 2}), t][1])(x)"

    leak_result = tracer_leak_autopsy(leak_code, topology_dict)
    assert "FAILED" in leak_result
    assert "leaky_param" in leak_result


def test_mcp_xla_interrogator():
    print("--- Testing MCP: XLA Memory Interrogator ---")

    topology_dict = {"q": [("s", 4096), ("d", 64)], "k": [("s", 4096), ("d", 64)]}

    # Simulating a massive attention dot-product (4096 x 4096)
    code = "q.d.proj(ax.d(64))"  # Simple projection to test the pipeline

    result = xla_memory_interrogator(code, topology_dict)

    assert "XLA Compilation Successful" in result
    assert "Largest Intermediate Buffer Allocated" in result


from axiom.mcp import prng_collision_detector, hardware_mesh_optimizer, associative_scan_profiler

def test_mcp_prng_detector():
    topology_dict = {"x": [("b", 1), ("s", 32), ("d", 64)]}
    code = "nn.dropout(x, rate=0.1, training=True)"
    result = prng_collision_detector(code, topology_dict)
    assert "PASSED" in result

def test_mcp_hardware_allocator():
    print("--- Testing MCP: Hardware Mesh Optimizer ---")
    # Testing our 180M model on an 8x A100-80GB GPU Node
    result = hardware_mesh_optimizer(180, 4096, "A100-80GBx8")

    assert "Pure Data Parallelism" in result
    assert "640" in result  # 80GB * 8 chips = 640GB total VRAM

def test_mcp_scan_profiler():
    result = associative_scan_profiler(4096, 256, 16)
    assert "Sub-optimal" in result


from axiom.mcp import compute_density_profiler, numerical_stability_oracle, optimal_remat_search


def test_mcp_compute_profiler():
    print("--- Testing MCP: Compute Density Profiler ---")
    topology_dict = {"x": [("b", 1), ("s", 128), ("d", 64)]}

    # A single linear projection
    code = "x.d.proj(ax.d(128))"
    result = compute_density_profiler(code, topology_dict)

    assert "Compute Density" in result
    assert "Total Parameters: 8,320" in result  # 64 * 128 + 128 bias
    # If the architecture is simple, it should calculate the ratio successfully
    assert "Compute-per-Neuron Ratio" in result


def test_mcp_stability_oracle():
    print("--- Testing MCP: Numerical Stability Oracle ---")
    topology_dict = {"x": [("b", 1), ("s", 32), ("d", 64)]}

    # Safe block (RMS Norm handles variance safely)
    safe_code = "x.d.rms_norm()"
    safe_result = numerical_stability_oracle(safe_code, topology_dict)
    assert "PASSED: Numerically stable" in safe_result


def test_mcp_remat_search():
    print("--- Testing MCP: Remat Search ---")

    # Test associative scan heuristic
    ssm_code = "(x & a).d.proj(ax.d(64))"
    result = optimal_remat_search(ssm_code)

    assert "associative scan" in result
    assert "Wrap the `(x & a)`" in result


from axiom.mcp import inspect_axiom_api


def test_mcp_api_inspector():
    print("--- Testing MCP: API Inspector ---")

    # 1. Test searching for a neural network layer
    nn_result = inspect_axiom_api("rms_norm")
    assert "Axiom API Reference:" in nn_result
    assert "rms_norm" in nn_result

    # 2. Test searching for a core tensor operation
    core_result = inspect_axiom_api("proj")
    assert "Source Code Preview:" in core_result

    # 3. Test a hallucinated function
    fake_result = inspect_axiom_api("make_me_a_sandwich")
    assert "❌ Could not find" in fake_result


from axiom.mcp import inspect_axiom_api, get_axiom_tutorial
import os


def test_mcp_api_inspector_deep_lookup():
    print("--- Testing MCP: API Inspector Deep Lookup ---")

    # Test method lookup inside a class (TargetedTensor.proj)
    # Based on our update, this should now work perfectly!
    proj_result = inspect_axiom_api("proj")
    assert "TargetedTensor.proj" in proj_result
    assert "Source Code Preview:" in proj_result


def test_mcp_tutorial_loader():
    print("--- Testing MCP: Tutorial Loader ---")

    # 1. Test success (assuming tutorial.py exists at examples/tutorial.py)
    # You may need to adjust the path if your structure is different
    tutorial_content = get_axiom_tutorial()

    if "❌ Could not find" not in tutorial_content:
        assert "Welcome to axiom!" in tutorial_content
        assert "def f(x: Tensor)" in tutorial_content
    else:
        print("Skipping tutorial content check: file not found in current path.")

    # 2. Test failure (hallucinated path if we renamed the file)
    # You could temporarily rename/move the file to verify the error string