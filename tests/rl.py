import jax.numpy as jnp
import optax
from axiom import ax, nn, init, Tensor

class Env:
    """Env: batched 2d discrete target seeker"""

    def __init__(self, batch_size=32):
        self.b = ax.b(batch_size)
        self.agent = init.zeros(self.b, ax.d(2)).unwrap()
        self.target = jnp.round(init.normal(self.b, ax.d(2)).unwrap() * 5.0)
        self.dist = jnp.linalg.norm(self.target - self.agent, axis=-1)

    def obs(self):
        """obs: returns relative distance vector"""
        return Tensor(self.target - self.agent, self.b, ax.d(2))

    def step(self, action: Tensor):
        """step: applies discrete action and calculates reward"""
        moves = jnp.array([[0, 1], [0, -1], [-1, 0], [1, 0], [0, 0]])
        self.agent += moves[action.unwrap()]
        self.agent = jnp.clip(self.agent, -10.0, 10.0)
        new_dist = jnp.linalg.norm(self.target - self.agent, axis=-1)
        is_parked = (new_dist < 0.1)
        reward = (self.dist - new_dist) + (is_parked * 0.2)
        self.dist = new_dist
        return self.obs(), Tensor(reward, self.b)

@ax.model
def actor(x: Tensor):
    """actor: outputs unnormalized logits for 5 discrete directions"""
    return x.d.proj(ax.h(32)).h.silu().h.proj(ax.a(5))

@ax.model
def critic(x: Tensor):
    """critic: evaluates state value, summing v(1) to return clean topology (b)"""
    return x.d.proj(ax.h(32)).h.silu().h.proj(ax.v(1)).v.sum()

actor_optim = optax.adam(1e-3)
critic_optim = optax.adam(1e-3)
actor_state = None
critic_state = None


@ax.jit
def train_step(actor, critic, act_state, crit_state, x, action, reward, next_x):
    def critic_loss(c_model):
        v = c_model(x)
        next_v = c_model(next_x).stop_grad()
        target = reward + next_v * 0.99
        return nn.mse_loss(v, target).b.mean()

    c_loss, c_grads = ax.value_and_grad(critic_loss)(critic)
    critic, crit_state = ax.apply_updates(critic, c_grads, critic_optim, crit_state)

    def actor_loss(a_model):
        logits = a_model(x)
        v = critic(x).stop_grad()
        next_v = critic(next_x).stop_grad()
        td_error = reward + next_v * 0.99 - v
        return nn.reinforce(logits.a, action, td_error).b.mean()

    a_loss, a_grads = ax.value_and_grad(actor_loss)(actor)
    actor, act_state = ax.apply_updates(actor, a_grads, actor_optim, act_state)

    return actor, critic, act_state, crit_state, a_loss, c_loss

env = Env()
x = env.obs()

for i in range(1000):
    logits = actor(x)
    action = logits.a.sample()
    next_x, reward = env.step(action)

    actor, critic, actor_state, critic_state, a_loss, c_loss = train_step(
        actor, critic, actor_state, critic_state, x, action, reward, next_x
    )

    print(f"loss: {c_loss:.4f} | dist: {env.dist.mean():.2f}")
    x = next_x