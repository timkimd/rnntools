import math

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import optax

import equinox as eqx
from equinox.nn.composed import _identity, Sequential

from findir.models import Recurrence, MyLinear, MyMLP, Dynamics_Decoder
from findir.generate_data import get_data
from findir.dataloader import dataloader

jax.config.update('jax_platform_name', 'cpu')

def train(
    dataset_size=10000,
    batch_size=32,
    learning_rate=1e-4,
    steps=300,
    hidden_size=16,
    seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)
    
    model = Dynamics_Decoder(
            alpha=1, 
            input_size=2,
            hidden_size=hidden_size,
            out_size=1,
            dynamics_width_size=100,
            dynamics_depth=1,
            gating_width_size=100,
            gating_depth=1,
            decoder_width_size=100,
            decoder_depth=1,
            sigma_w=1,
            hidden_activation=jnn.relu,
            dynamics_activation=jnn.tanh,
            gating_activation=jnn.sigmoid,
            key=model_key
        )

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        model_y = jax.vmap(model)(x)
        pred_y = jnn.sigmoid(model_y)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")