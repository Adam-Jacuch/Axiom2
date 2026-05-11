import pytest
import jax
import jax.numpy as jnp
import optax
from axiom import ax, Tensor, wrap, axiom_jit, axiom_step, nn


def test_end_to_end_training_step():
    # 1. Define the network
    @axiom_jit
    def simple_mlp(x: Tensor):
        # We can dynamically declare axis sizes right in the execution if we want!
        hidden = x.in_d.proj(ax.out_d(1)).pw(nn.sigmoid)
        return hidden

    # 2. Define the optimizer
    optimizer = optax.adam(learning_rate=0.1)

    # 3. Define the training loop
    @axiom_step(model=simple_mlp, optimizer=optimizer)
    def train_step(x: Tensor, y_true: Tensor):
        preds = simple_mlp(x)

        # Loss computation
        loss = nn.mse_loss(preds, y_true).b.mean()

        # Backward pass & Optax Update
        simple_mlp.vjp(loss)
        simple_mlp.step()

        return loss

    # Inline axis initialization during array wrapping!
    x_data = wrap(jnp.ones((8, 16)), ax.b(8), ax.in_d(16))
    y_target = wrap(jnp.ones((8, 1)), ax.b(8), ax.out_d(1))

    # Record the initial loss
    initial_loss = train_step(x_data, y_target)

    # Train for 10 steps
    for _ in range(10):
        final_loss = train_step(x_data, y_target)

    assert final_loss < initial_loss

    state = simple_mlp.get_state()
    assert "param_0" in state