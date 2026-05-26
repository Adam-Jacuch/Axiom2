import jax
import jax.numpy as jnp
import optax
from axiom import ax, nn, init, Tensor

@jax.tree_util.register_pytree_node_class
class Env:
    """Env: batched 2d discrete target seeker"""
    def __init__(self, batch_size=32):
        self.b = ax.b(batch_size)
        self.agent = init.zeros(self.b, ax.d(2)).unwrap()
        self.target = jnp.round(init.normal(self.b, ax.d(2)).unwrap() * 5.0)
        self.dist = jnp.linalg.norm(self.target - self.agent, axis=-1)

    def tree_flatten(self):
        children = (self.agent, self.target, self.dist)
        aux_data = (self.b,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.agent, obj.target, obj.dist = children
        obj.b = aux_data[0]
        return obj

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
def train_step(actor, critic, act_state, crit_state, env):
    data = []
    x = env.obs()
    v = critic(x).stop_grad()

    for j in range(100):
        a = actor(x).stop_grad().a.sample()
        next_x, r = env.step(a)
        v2 = critic(next_x).stop_grad()

        data.append({"x": x, "action": a, "value": v, "next_value": v2, "reward": r})
        x = next_x
        v = v2

    gae = 0.0
    for d in reversed(data):
        delta = d["reward"] + d["next_value"] * 0.99 - d["value"]
        gae = delta + gae * (0.99 * 0.95)
        d["advantage"] = gae

    time = ax.t(100)
    inputs = ax.stack([d["x"] for d in data], time)
    advantages = ax.stack([d["advantage"] for d in data], time)
    values = ax.stack([d["value"] for d in data], time)
    returns = advantages + values
    actions = ax.stack([d["action"] for d in data], time)

    def critic_loss(c_model):
        v_preds = c_model(inputs)
        return nn.mse_loss(v_preds, returns)

    def actor_loss(a_model):
        logits = a_model(inputs)
        entropy = logits.a.pw(lambda v: jax.nn.softmax(v) * jax.nn.log_softmax(v)).a.sum().b.t.mean()
        return nn.reinforce(logits.a, actions, advantages) + entropy * 0.01

    c_loss, c_grads = ax.value_and_grad(critic_loss)(critic)
    a_loss, a_grads = ax.value_and_grad(actor_loss)(actor)

    critic, crit_state = ax.apply_updates(critic, c_grads, critic_optim, crit_state)
    actor, act_state = ax.apply_updates(actor, a_grads, actor_optim, act_state)

    return actor, critic, act_state, crit_state, a_loss, c_loss, env

env = Env()
for i in range(1000):
    actor, critic, actor_state, critic_state, a_loss, c_loss, env = train_step(
        actor, critic, actor_state, critic_state, env
    )
    print(f"loss: {c_loss:.4f} | dist: {env.dist.mean():.2f}")