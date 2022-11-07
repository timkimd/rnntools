import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array

import equinox as eqx
from equinox.module import static_field
from equinox.nn.composed import _identity

jax.config.update('jax_platform_name', 'cpu')

class MyLinear(eqx.Module):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: int = static_field()
    out_features: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        sigma_w: float = 1,
        *,
        key: "jax.random.PRNGKey"
    ):
        """**Arguments:**
        - `in_features`: The input size.
        - `out_features`: The output size.
        - `use_bias`: Whether to add on a bias as well.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        wkey, bkey = jrandom.split(key, 2)
        self.weight = sigma_w / math.sqrt(in_features) * jrandom.normal(
            wkey, (out_features, in_features)
        )
        if use_bias:
            self.bias = 1e-3 * jrandom.normal(
                bkey, (out_features,)
            )
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**
        - `x`: The input. Should be a JAX array of shape `(in_features,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        !!! info
            If you want to use higher order tensors as inputs (for example featuring batch dimensions) then use
            `jax.vmap`. For example, for an input `x` of shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.
        **Returns:**
        A JAX array of shape `(out_features,)`
        """

        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x

class MyMLP(eqx.Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""

    layers: List[MyLinear]
    activation: Callable
    final_activation: Callable
    in_size: int = static_field()
    out_size: int = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        sigma_w: float = 1,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments**:
        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(MyLinear(in_size, out_size, sigma_w=sigma_w, key=keys[0]))
        else:
            layers.append(MyLinear(in_size, width_size, sigma_w=sigma_w, key=keys[0]))
            for i in range(depth - 1):
                layers.append(MyLinear(width_size, width_size, sigma_w=sigma_w, key=keys[i + 1]))
            layers.append(MyLinear(width_size, out_size, sigma_w=sigma_w, key=keys[-1]))
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """**Arguments:**
        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array with shape `(out_size,)`.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x

class NeuralODECell(eqx.Module):
    """A single step of a Neural ODE (nODE) with Euler steps.
    !!! example
        This is often used by wrapping it into a `jax.lax.scan`. For example:
        ```python
        class Model(Module):
            cell: NeuralODECell
            def __init__(self, **kwargs):
                self.cell = NeuralODECell(**kwargs)
            def __call__(self, xs):
                scan_fn = lambda state, input: (self.cell(input, state), None)
                init_state = jnp.zeros(self.cell.hidden_size)
                final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
                return final_state
        ```
    """

    weight_ih: Array
    weight_hh: Array
    bias: Optional[Array]
    f: Optional[eqx.Module]
    activation: Callable
    final_activation: Callable

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        sigma_w: float = 1,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        *,
        key: Optional["jax.random.PRNGKey"],
        **kwargs
    ):
        """**Arguments:**
        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(**kwargs)

        if depth == 0:
            subkeys = jrandom.split(key, 3)

            self.weight_ih = sigma_w / math.sqrt(input_size) * jrandom.normal(
                subkeys[-1], (hidden_size, input_size)
            )
            self.weight_hh = sigma_w / math.sqrt(hidden_size) * jrandom.normal(
                subkeys[-2], (hidden_size, hidden_size)
            )
            self.bias = 1e-3 * jrandom.normal(
                subkeys[-3], (hidden_size,)
            )
            self.f = None
        else:
            subkeys = jrandom.split(key, 4)

            self.weight_ih = sigma_w / math.sqrt(input_size) * jrandom.normal(
                subkeys[-1], (width_size, input_size)
            )
            self.weight_hh = sigma_w / math.sqrt(hidden_size) * jrandom.normal(
                subkeys[-2], (width_size, hidden_size)
            )
            self.bias = 1e-3 * jrandom.normal(
                subkeys[-3], (width_size,)
            )
            self.f = MyMLP(
                width_size,
                hidden_size,
                width_size,
                depth-1,
                activation,
                final_activation,
                sigma_w,
                key = subkeys[-4]
            )
        self.activation = activation
        self.final_activation = final_activation

    def __call__(
        self, input: Array, hidden: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ):
        """**Arguments:**
        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a JAX array of shape
            `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        The updated hidden state, which is a JAX array of shape `(hidden_size,)`.
        """

        h_tilde = self.weight_ih @ input + self.weight_hh @ hidden + self.bias
        if self.f is None:
            return -hidden + self.final_activation(h_tilde)
        else:
            return -hidden + self.f(self.activation(h_tilde))

class GatedNeuralODECell(eqx.Module):
    """A single step of a Gated Neural ODE (gnODE) with Euler steps.
    !!! example
        This is often used by wrapping it into a `jax.lax.scan`. For example:
        ```python
        class Model(Module):
            cell: GatedNeuralODECell
            def __init__(self, **kwargs):
                self.cell = GatedNeuralODECell(**kwargs)
            def __call__(self, xs):
                scan_fn = lambda state, input: (self.cell(input, state), None)
                init_state = jnp.zeros(self.cell.hidden_size)
                final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
                return final_state
        ```
    """

    weight_ih: Array
    weight_hh: Array
    weight_iz: Array
    weight_hz: Array
    bias_h: Optional[Array]
    bias_z: Optional[Array]
    f: Optional[eqx.Module]
    g: Optional[eqx.Module]
    hidden_activation: Callable
    dynamics_activation: Callable
    gating_activation: Callable

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dynamics_width_size: int,
        dynamics_depth: int,
        gating_width_size: int,
        gating_depth: int,
        sigma_w: float = 1,
        hidden_activation: Callable = jnn.relu,
        dynamics_activation: Callable = jnn.tanh,
        gating_activation: Callable = _identity,
        *,
        key: Optional["jax.random.PRNGKey"],
        **kwargs
    ):
        """**Arguments:**
        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(**kwargs)

        subkeys = jrandom.split(key, 6 + bool(dynamics_depth) + bool(gating_depth))
        if dynamics_depth == 0:
            self.weight_ih = sigma_w / math.sqrt(input_size) * jrandom.normal(
                subkeys[-1], (hidden_size, input_size)
            )
            self.weight_hh = sigma_w / math.sqrt(hidden_size) * jrandom.normal(
                subkeys[-2], (hidden_size, hidden_size)
            )
            self.bias_h = 1e-3 * jrandom.normal(
                subkeys[-3], (hidden_size,)
            )
            self.f = None
        else:
            self.weight_ih = sigma_w / math.sqrt(input_size) * jrandom.normal(
                subkeys[-1], (dynamics_width_size, input_size)
            )
            self.weight_hh = sigma_w / math.sqrt(hidden_size) * jrandom.normal(
                subkeys[-2], (dynamics_width_size, hidden_size)
            )
            self.bias_h = 1e-3 * jrandom.normal(
                subkeys[-3], (dynamics_width_size,)
            )
            self.f = MyMLP(
                dynamics_width_size,
                hidden_size,
                dynamics_width_size,
                dynamics_depth-1,
                hidden_activation,
                dynamics_activation,
                sigma_w,
                key = subkeys[-4]
            )
        
        if gating_depth == 0:
            self.weight_iz = sigma_w / math.sqrt(input_size) * jrandom.normal(
                subkeys[0], (hidden_size, input_size)
            )
            self.weight_hz = sigma_w / math.sqrt(hidden_size) * jrandom.normal(
                subkeys[1], (hidden_size, hidden_size)
            )
            self.bias_z = 1e-3 * jrandom.normal(
                subkeys[2], (hidden_size,)
            )
            self.g = None
        else:
            self.weight_iz = sigma_w / math.sqrt(input_size) * jrandom.normal(
                subkeys[0], (gating_width_size, input_size)
            )
            self.weight_hz = sigma_w / math.sqrt(hidden_size) * jrandom.normal(
                subkeys[1], (gating_width_size, hidden_size)
            )
            self.bias_z = 1e-3 * jrandom.normal(
                subkeys[2], (gating_width_size,)
            )
            self.g = MyMLP(
                gating_width_size,
                hidden_size,
                gating_width_size,
                gating_depth-1,
                hidden_activation,
                gating_activation,
                sigma_w,
                key = subkeys[3]
            )
        
        self.hidden_activation = hidden_activation
        self.dynamics_activation = dynamics_activation
        self.gating_activation = gating_activation

    def __call__(
        self, input: Array, hidden: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ):
        """**Arguments:**
        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a JAX array of shape
            `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        The updated hidden state, which is a JAX array of shape `(hidden_size,)`.
        """

        z_tilde = self.weight_iz @ input + self.weight_hz @ hidden + self.bias_z
        h_tilde = self.weight_ih @ input + self.weight_hh @ hidden + self.bias_h
        if self.g is None:
            z_tilde = self.gating_activation(z_tilde)
        else:
            z_tilde = self.g(self.hidden_activation(z_tilde))

        if self.f is None:
            h_tilde = self.dynamics_activation(h_tilde)
        else:
            h_tilde = self.f(self.hidden_activation(h_tilde))
        return z_tilde * (-hidden + h_tilde)

class Recurrence(eqx.Module):
    cell: eqx.Module
    alpha: float = static_field()
    hidden_size: int = static_field()

    def __init__(
        self, 
        alpha: float,
        input_size: int,
        hidden_size: int,
        dynamics_width_size: int,
        dynamics_depth: int,
        gating_width_size: int,
        gating_depth: int,
        sigma_w: float = 1,
        hidden_activation: Callable = jnn.relu,
        dynamics_activation: Callable = jnn.tanh,
        gating_activation: Callable = jnn.sigmoid,
        *, 
        key
    ):
        self.alpha = alpha
        self.hidden_size = hidden_size
        if gating_depth < 0:
            self.cell = NeuralODECell(
                input_size, 
                hidden_size, 
                dynamics_width_size,
                dynamics_depth, 
                sigma_w, 
                hidden_activation,
                dynamics_activation,
                key=key
            )
        else:
            self.cell = GatedNeuralODECell(
                input_size, 
                hidden_size, 
                dynamics_width_size,
                dynamics_depth,
                gating_width_size,
                gating_depth, 
                sigma_w, 
                hidden_activation,
                dynamics_activation,
                gating_activation,
                key=key
            )        

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            h_tilde = self.cell(inp, carry)
            h = carry + self.alpha * h_tilde
            return h, None

        out, _ = lax.scan(f, hidden, input)
        return out

class Dynamics_Decoder(eqx.Module):
    dynamics_model: eqx.Module
    decoder: eqx.nn.MLP

    def __init__(
        self,
        alpha: float,
        input_size: int,
        hidden_size: int,
        out_size:int,
        dynamics_width_size: int,
        dynamics_depth: int,
        gating_width_size: int,
        gating_depth: int,
        decoder_width_size: int,
        decoder_depth: int,
        sigma_w: float = 1,
        hidden_activation: Callable = jnn.relu,
        dynamics_activation: Callable = jnn.tanh,
        gating_activation: Callable = jnn.sigmoid,
        *, 
        key
    ):
        dynamics_model_key, decoder_key = jrandom.split(key, 2)
        self.dynamics_model = Recurrence(
            alpha,
            input_size,
            hidden_size,
            dynamics_width_size,
            dynamics_depth,
            gating_width_size,
            gating_depth,
            sigma_w,
            hidden_activation,
            dynamics_activation,
            gating_activation,
            key = dynamics_model_key
        )
        self.decoder = eqx.nn.MLP(
            hidden_size,
            out_size,
            decoder_width_size,
            decoder_depth,
            key = decoder_key
        )
    
    def __call__(self, x):
        h = self.dynamics_model(x)
        y = self.decoder(h)
        return y