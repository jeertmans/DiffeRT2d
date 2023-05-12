from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from chex import Array

jit = jax.jit
jit_approx = partial(jax.jit, static_argnames=["approx"])


@jit
def sigmoid(x: Array, lambda_=50.0, **kwargs) -> Array:
    # https://stackoverflow.com/questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan
    # return 0.5 * (jnp.tanh(x * lambda_ / 2) + 1)
    return jax.nn.sigmoid(lambda_ * x)


@jit_approx
def true_(approx: bool = True, **kwargs) -> Union[bool, float]:
    return 1.0 if approx else True


@jit_approx
def false_(approx: bool = True, **kwargs) -> Union[bool, float]:
    return 0.0 if approx else False


@jit_approx
def or_(x, y, approx=True, **kwargs):
    return jnp.maximum(x, y) if approx else jnp.logical_or(x, y)


@jit_approx
def and_(x, y, approx=True, **kwargs):
    return x * y if approx else jnp.logical_and(x, y)


@jit_approx
def not_(x, approx=True, **kwargs):
    return 1 - x if approx else jnp.logical_not(x)


@jit_approx
def ne_(x, y, approx=True, **kwargs):
    return jnp.abs(x - y) if approx else jnp.not_equal(x, y)


@jit_approx
def eq_(x, y, approx=True, **kwargs):
    return 1 - jnp.abs(x - y) if approx else jnp.equal(x, y)


@jit_approx
def gt_(x, y, approx=True, **kwargs):
    return sigmoid(x - y, **kwargs) if approx else jnp.greater(x, y)
    # return jnp.greater(x, y) * 1.0 if approx else jnp.greater(x, y)


@jit_approx
def lt_(x, y, approx=True, **kwargs):
    return sigmoid(y - x, **kwargs) if approx else jnp.less(x, y)


@jit_approx
def is_pos(x, **kwargs):
    return gt_(x, 0, **kwargs)
