import equinox as eqx
import functools as ft
import jax

jax.config.update('jax_platform_name', 'cpu')

class AnotherModule(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jax.random.split(key)
        self.layers = [eqx.nn.Linear(2, 8, key=key1),
                       jax.nn.relu,
                       eqx.nn.Linear(8, 2, key=key2)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
x, y = jax.random.normal(x_key, (100, 2)), jax.random.normal(y_key, (100, 2))
model = AnotherModule(model_key)

