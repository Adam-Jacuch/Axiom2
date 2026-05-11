import jax
import jax.numpy as jnp
import optax
from axiom import ax, Tensor, wrap, axiom_jit, axiom_step, nn


# ==========================================
# 1. THE AXIOM POLICY NETWORK (ACTOR)
# ==========================================
@axiom_jit
def policy_network(state: Tensor):
    """Predicts the mean (mu) of the optimal action distribution."""
    # Notice the inline axis sizes!
    h = state.state_d.proj(ax.hidden_d(32)).pw(nn.relu)
    mu = h.hidden_d.proj(ax.action_d(2))
    return mu


# ==========================================
# 2. THE DISTRIBUTED TRAINING ENGINE
# ==========================================
optimizer = optax.adam(learning_rate=0.05)


@axiom_step(model=policy_network, optimizer=optimizer)
def train_step(state: Tensor, taken_action: Tensor, reward: Tensor):
    """Pure functional REINFORCE Policy Gradient."""
    mu = policy_network(state)

    # 1. Calculate the Log Probability of the action taken
    # Assuming a Gaussian policy with variance = 1: log_prob ∝ -0.5 * (x - mu)^2
    log_prob = (taken_action - mu).pw(jnp.square) * -0.5

    # Subtract the mean reward to create an Advantage!
    # Axiom natively broadcasts the scalar mean across the batch!
    advantage = reward - reward.b.mean()

    # 2. Policy Gradient Objective
    # We want to maximize expected reward, which means minimizing: -(log_prob * reward)
    # Axiom natively broadcasts the 'b' axis of the reward across the action dimensions!
    loss = -(log_prob * advantage).b.action_d.mean()

    # 3. Syntactic Sugar
    policy_network.vjp(loss)
    policy_network.step()

    return loss


# ==========================================
# 3. THE RL ENVIRONMENT LOOP
# ==========================================
def run_training():
    print("Initializing Axiom Continuous RL Environment...")
    key = jax.random.PRNGKey(42)
    batch_size = 128

    # Define the static hidden rules of the environment
    env_key = jax.random.PRNGKey(99)
    W_env_raw = jax.random.normal(env_key, (4, 2))
    W_env = wrap(W_env_raw, ax.state_d(4), ax.action_d(2))

    for epoch in range(1, 2001):
        key, s_key, a_key = jax.random.split(key, 3)

        # 1. Environment: Generate Random States
        raw_states = jax.random.normal(s_key, (batch_size, 4))
        states = wrap(raw_states, ax.b(batch_size), ax.state_d(4))

        # 2. Agent: Choose Actions (Exploration via Gaussian Noise)
        # We can call our compiled model eagerly to get the mean!
        mu_tensor = policy_network(states)

        noise_raw = jax.random.normal(a_key, (batch_size, 2))
        noise_tensor = wrap(noise_raw, ax.b(batch_size), ax.action_d(2))

        # Axiom Native Math
        taken_actions = mu_tensor + noise_tensor

        # 3. Environment: Compute Reward
        # Pure Axiom Math: Multiply the state by the hidden environment matrix and sum!
        optimal_actions = (states * W_env).state_d.sum()

        # Reward is the negative Euclidean distance to the optimal action
        distances = (taken_actions - optimal_actions).pw(jnp.square).action_d.sum().pw(jnp.sqrt)
        rewards = -distances

        # 4. Agent: Learn from the Experience
        loss = train_step(states, taken_actions, rewards)

        # Print progress
        if epoch % 20 == 0:
            avg_reward = rewards.b.mean().unwrap()
            print(f"Epoch {epoch:03d} | Avg Reward: {avg_reward:7.3f} | Loss: {loss:7.3f}")

    print("\nTraining Complete! The Axiom engine successfully optimized the policy.")


if __name__ == "__main__":
    run_training()