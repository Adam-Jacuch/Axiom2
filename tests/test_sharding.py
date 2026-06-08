import os

# MUST BE SET BEFORE JAX IMPORTS!
# This tricks your laptop into simulating a 4-chip cluster.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
from axiom import ax, init, Tensor
import numpy as np


def test_cpu_fsdp_sharding_simulation():
    print(f"\\n--- Simulating Cluster: {jax.device_count()} CPU Devices ---")

    # 1. Verify JAX successfully faked the hardware
    assert jax.device_count() == 4, "JAX did not emulate the 4 devices!"

    # 2. Define a simple mathematical block
    def fsdp_block(x: Tensor):
        # A standard projection that we want to shard
        return x.d.proj(ax.h(128))

    # 3. Compile with the new Axiom Sharding Bridge (FSDP - 2D Mesh)
    @ax.jit(shard=[ax.b, ax.d])
    def step(model, x):
        return model(x)

    # 4. Setup dummy axes and inputs
    b, s, d = ax.b(8), ax.s(16), ax.d(64)
    x = init.normal(b, s, d)

    # 5. Initialize the functional model
    model = ax.model(fsdp_block).init(b, s, d)

    # 6. Execute! (Axiom will build the 2x2 mesh and shard the arrays)
    out = step(model, x)

    # 7. THE PROOF: Inspect the underlying JAX array
    jax_array = out.unwrap()

    print(f"Output Shape: {jax_array.shape}")
    print(f"Output Sharding Spec: {jax_array.sharding}")

    # If the bridge worked, the output array will be governed by a NamedSharding rule
    assert "NamedSharding" in str(type(jax_array.sharding))

    # JAX arrays also expose whether they are fully replicated or physically sliced
    assert not jax_array.is_fully_replicated, "Error: Tensor is living entirely on one device!"

    print("✅ Sharding Bridge validated! Tensor is physically shattered across the CPU cluster.")