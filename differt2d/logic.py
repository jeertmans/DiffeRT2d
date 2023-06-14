"""
A toolbox for logical operations.

When approximation is enabled, a value close to 1 maps to `True`,
while a value close to 0 maps `False`.

Otherwise, functions will call their JAX counterpart. E.g.,
:func:`logical_or` calls :func:`jax.numpy.logical_or`
when :code:`approx` is set to :code:`False`.
"""

__all__ = [
    "disable_approx",
    "enable_approx",
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

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array
else:
    Array = Any

_enable_approx = jax.config.define_bool_state(
    name="jax_enable_approx",
    default=True,
    help=("Enable approximation using sigmoids."),
)

jit_approx = partial(jax.jit, inline=True, static_argnames=["approx"])


@contextmanager
def enable_approx(enable: bool = True):
    """
    Context manager for enabling or disabling approximation of true/false values
    with continuous numbers from 0 (false) to 1 (true).

    By default, approximation is enabled.

    To disable approximation, you have multiple options:

    1. use this context manager to disable it (see example below);
    2. set the environ variable ``JAX_ENABLE_APPROX`` to ``0``
       (or any falsy value);
    3. update the config with ``jax.config.update("jax_enable_approx", False)``;
    4. or set, for specific logic functions only, the keyword argument
       ``approx`` to ``False``.

    :Examples:

    >>> from differt2d.logic import enable_approx, greater
    >>>
    >>> greater.clear_caches()  # doc: hide
    >>> with enable_approx(False):
    ...     print(greater(20.0, 5.0))
    True

    You can also enable approximation with this:

    >>> greater.clear_caches()  # doc: hide
    >>> with enable_approx(True):
    ...     print(greater(20.0, 5.0))
    1.0

    Calling without args defaults to True:

    >>> greater.clear_caches()  # doc: hide
    >>> with enable_approx():
    ...     print(greater(20.0, 5.0))
    1.0

    .. warning::

        Calling already-jitted functions after mutating ``jax_enable_approx``
        will not produce any visible change. This is because
        ``jax.config.jax_enable_approx`` is evaluated once, at compilation.

        For example:

        >>> import jax
        >>> from differt2d.logic import enable_approx
        >>>
        >>> @jax.jit
        ... def f():
        ...     if jax.config.jax_enable_approx:
        ...         return 1.0
        ...     else:
        ...         return 0.0
        >>>
        >>> with enable_approx(True):
        ...     print(f())
        1.0
        >>>
        >>> with enable_approx(False):
        ...     print(f())
        1.0

        To avoid this issue, you can either disable jit with
        :py:func:`jax.disable_jit` or use the ``approx`` parameter
        when available.

        >>> from jax import disable_jit
        >>>
        >>> @jax.jit
        ... def f():
        ...     if jax.config.jax_enable_approx:
        ...         return 1.0
        ...     else:
        ...         return 0.0
        >>>
        >>> with enable_approx(True), disable_jit():
        ...     print(f())
        1.0
        >>>
        >>> with enable_approx(False), disable_jit():
        ...     print(f())
        0.0
    """
    with _enable_approx(enable):
        yield


@contextmanager
def disable_approx(disable: bool = True):
    """
    Context manager for disable or enabling approximation of true/false values
    with continuous numbers from 0 (false) to 1 (true).

    This function is an alias to ``enable_approx(not disable)``.
    For more details, refer to :py:func:`enable_approx`.

    .. note::

        Contrary to ``enable_approx``, there is no ``JAX_DISABLE_APPROX``
        environ variable, nor ``jax.config.jax_disable_approx`` config variable.
    """
    with _enable_approx(not disable):
        yield


@partial(jax.jit, inline=True)
def sigmoid(x: Array, *, lambda_: float = 100.0) -> Array:
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
def logical_or(x: Array, y: Array, *, approx: Optional[bool] = None) -> Array:
    """
    Element-wise logical OR operation betwen :code:`x` and :code:`y`.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.maximum(x, y) if approx else jnp.logical_or(x, y)


@jit_approx
def logical_and(x: Array, y: Array, *, approx: Optional[bool] = None) -> Array:
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.multiply(x, y) if approx else jnp.logical_and(x, y)


@jit_approx
def logical_not(x: Array, *, approx: Optional[bool] = None) -> Array:
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.subtract(1.0, x) if approx else jnp.logical_not(x)


@jit_approx
def greater(
    x: Array,
    y: Array,
    *,
    approx: Optional[bool] = None,
    **kwargs,
) -> Array:
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return sigmoid(jnp.subtract(x, y), **kwargs) if approx else jnp.greater(x, y)


@jit_approx
def greater_equal(
    x: Array, y: Array, *, approx: Optional[bool] = None, **kwargs
) -> Array:
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return sigmoid(jnp.subtract(x, y), **kwargs) if approx else jnp.greater_equal(x, y)


@jit_approx
def less(x: Array, y: Array, *, approx: Optional[bool] = None, **kwargs) -> Array:
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return sigmoid(jnp.subtract(y, x), **kwargs) if approx else jnp.less(x, y)


@jit_approx
def less_equal(x: Array, y: Array, *, approx: Optional[bool] = None, **kwargs) -> Array:
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return sigmoid(jnp.subtract(y, x), **kwargs) if approx else jnp.less_equal(x, y)


@jit_approx
def is_true(x: Array, *, tol: float = 0.5, approx: Optional[bool] = None) -> Array:
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
    :type approx: typing.Optional[bool]
    :return: True if the value is considered to be true.
    :rtype: jax.Array
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.greater(x, 1.0 - tol) if approx else jnp.asarray(x)


@jit_approx
def is_false(x: Array, *, tol: float = 0.5, approx: Optional[bool] = None) -> Array:
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
    :type approx: typing.Optional[bool]
    :return: True if the value is considered to be false.
    :rtype: jax.Array
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.less(x, tol) if approx else jnp.logical_not(x)
