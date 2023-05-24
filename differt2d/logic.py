"""
A toolbox for logical operations.

When approximation is enabled, a value close to 1 maps to `True`,
while a value close to 0 maps `False`.

Otherwise, functions will call their JAX counterpart. E.g.,
:func:`logical_or` calls :func:`jax.numpy.logical_or`
when :code:`approx` is set to :code:`False`.
"""

__all__ = [
    "APPROX",
    "greater",
    "greater_equal",
    "is_false",
    "is_true",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "sigmoid",
]

from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array
else:
    Array = Any

jit_approx = partial(jax.jit, inline=True, static_argnames=["approx"])

APPROX = False
"""
Sets the default :code:`approx` keyword argument for all functions.
"""


@partial(jax.jit, inline=True)
def sigmoid(x: Array, lambda_: float = 100.0, **kwargs) -> Array:
    """
    Element-wise function for approximating a discrete transition between 0 and 1,
    with a smoothed transition.

    .. math::
        \\text{sigmoid}(x;\\lambda) = \\frac{1}{1 + e^{-\\lambda x}},

    where :math:`\\lambda` (:code:`lambda_`) is a slope parameter.

    See :func:`jax.nn.sigmoid` for more details.

    .. note::

        Using the above definition for the sigmoid will produce
        undesirable effects when computing its gradient. This is why we rely
        on JAX's implementation, that does not produce :code:`NaN` values
        when :code:`x` is small.

        You can read more about this in
        :sothread:`questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan`.

    :param x: The input array.
    :type x: jax.Array
    :param `lambda_`: The slope parameter.
    :type `lambda_`: float
    :return: The corresponding sigmoid values.
    :rtype: jax.Array
    """
    # https://stackoverflow.com/questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan
    # return 0.5 * (jnp.tanh(x * lambda_ / 2) + 1)
    return jax.nn.sigmoid(lambda_ * x)


@jit_approx
def logical_or(x: Array, y: Array, approx: bool = APPROX) -> Array:
    """
    Element-wise logical OR operation betwen :code:`x` and :code:`y`.
    """
    return jnp.maximum(x, y) if approx else jnp.logical_or(x, y)


@jit_approx
def logical_and(x: Array, y: Array, approx: bool = APPROX) -> Array:
    return jnp.multiply(x, y) if approx else jnp.logical_and(x, y)


@jit_approx
def logical_not(x: Array, approx: bool = APPROX) -> Array:
    return jnp.subtract(1.0, x) if approx else jnp.logical_not(x)


@jit_approx
def greater(x: Array, y: Array, approx: bool = APPROX) -> Array:
    return sigmoid(jnp.subtract(x, y)) if approx else jnp.greater(x, y)


@jit_approx
def greater_equal(x: Array, y: Array, approx: bool = APPROX) -> Array:
    return sigmoid(jnp.subtract(x, y)) if approx else jnp.greater_equal(x, y)


@jit_approx
def less(x: Array, y: Array, approx: bool = APPROX) -> Array:
    return sigmoid(jnp.subtract(y, x)) if approx else jnp.less(x, y)


@jit_approx
def less_equal(x: Array, y: Array, approx: bool = APPROX) -> Array:
    return sigmoid(jnp.subtract(y, x)) if approx else jnp.less_equal(x, y)


@jit_approx
def is_true(x: Array, tol: float = 0.5, approx: bool = APPROX) -> Array:
    """
    Element-wise check if a given truth value can be considered to be true.

    When using approximation,
    this function checks whether the value is close to 1.

    :param x: A truth array.
    :type x: jax.Array
    :param tol: The tolerance on how close it should be to 1.
        Only used if :code:`approx` is set to :code:`True`.
    :type tol: float
    :param approx: Whether approximation is used or not.
    :type approx: bool
    :return: True if the value is considered to be true.
    :rtype: jax.Array
    """
    return jnp.greater(x, 1.0 - tol) if approx else x


@jit_approx
def is_false(x: Array, tol: float = 0.5, approx: bool = APPROX) -> Array:
    """
    Element-wise check if a given truth value can be considered to be false.

    When using approximation,
    this function checks whether the value is close to 0.

    :param x: A truth array.
    :type x: jax.Array
    :param tol: The tolerance on how close it should be to 0.
        Only used if :code:`approx` is set to :code:`True`.
    :type tol: float
    :param approx: Whether approximation is used or not.
    :type approx: bool
    :return: True if the value is considered to be false.
    :rtype: jax.Array
    """
    return jnp.less(x, tol) if approx else jnp.logical_not(x)
