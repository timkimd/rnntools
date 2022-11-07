import math

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax  # https://github.com/deepmind/optax

import equinox as eqx

jax.config.update('jax_platform_name', 'cpu')