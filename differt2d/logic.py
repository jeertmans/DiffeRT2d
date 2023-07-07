"""
A toolbox for logical operations.

When approximation is enabled, a value close to 1 maps to :python:`True`,
while a value close to 0 maps :python:`False`.

Otherwise, functions will call their JAX counterpart. E.g.,
:py:func:`logical_or` calls :py:func:`jax.numpy.logical_or`
when :code:`approx` is set to :python:`False`.

.. note::

    Whenever a function takes an argument named ``approx``, it can take
    three different values:

    1. :python:`None`: defaults to :py:attr:`jax.config.jax_enable_approx`,
       see :py:func:`enable_approx` for comments on that;
    2. :python:`True`: forces to enable approximation;
    3. or :python:`False`: forces to disable approximation.
"""

from __future__ import annotations

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
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

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
    3. update the config with
       :python:`jax.config.update("jax_enable_approx", False)`;
    4. or set, for specific logic functions only, the keyword argument
       ``approx`` to :python:`False`.

    :param enable: Whether to enable or not approximation.

    :Examples:

    >>> from differt2d.logic import enable_approx, greater
    >>>
    >>> # doc: hide
    >>> greater.clear_cache()
    >>> # doc: hide
    >>> with enable_approx(False):
    ...     print(greater(20.0, 5.0))
    True

    You can also enable approximation with this:

    >>> # doc: hide
    >>> greater.clear_cache()
    >>> # doc: hide
    >>> with enable_approx(True):
    ...     print(greater(20.0, 5.0))
    1.0

    Calling without args defaults to True:

    >>> # doc: hide
    >>> greater.clear_cache()
    >>> # doc: hide
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

    This function is an alias to :python:`enable_approx(not disable)`.
    For more details, refer to :py:func:`enable_approx`.

    :param disable: Whether to disable or not approximation.

    .. note::

        Contrary to :py:func:`enable_approx`, there is no ``JAX_DISABLE_APPROX``
        environ variable, nor ``jax.config.jax_disable_approx`` config variable.
    """
    with _enable_approx(not disable):
        yield


@partial(jax.jit, inline=True)
def sigmoid(x: Array, *, lambda_: float = 100.0) -> Array:
    r"""
    Element-wise function for approximating a discrete transition between 0 and 1,
    with a smoothed transition.

    .. math::
        \text{sigmoid}(x;\lambda) = \frac{1}{1 + e^{-\lambda x}},

    where :math:`\lambda` (:code:`lambda_`) is a slope parameter.

    See :func:`jax.nn.sigmoid` for more details.

    .. note::

        Using the above definition for the sigmoid will produce
        undesirable effects when computing its gradient. This is why we rely
        on JAX's implementation, that does not produce :code:`NaN` values
        when :code:`x` is small.

        You can read more about this in
        :sothread:`questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan`.

    :param x: The input array.
    :param `lambda_`: The slope parameter.
    :return: The corresponding sigmoid values.

    :EXAMPLES:

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import numpy as np
        from differt2d.logic import sigmoid

        x = np.linspace(-5, +5, 200)

        for lambda_ in [1, 10, 100]:
            y = sigmoid(x, lambda_=lambda_)
            _ = plt.plot(x, y, "--", label=f"$\\lambda = {lambda_}$")

        plt.xlabel("$x$")
        plt.ylabel(r"sigmoid$(x;\lambda)$")
        plt.legend()
        plt.show()
    """
    return jax.nn.sigmoid(lambda_ * x)


@jit_approx
def logical_or(x: Array, y: Array, *, approx: Optional[bool] = None) -> Array:
    """
    Element-wise logical :python:`x or y`.

    Calls :func:`jax.numpy.maximum` if approximation is enabled,
    :func:`jax.numpy.logical_or` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.maximum(x, y) if approx else jnp.logical_or(x, y)


@jit_approx
def logical_and(x: Array, y: Array, *, approx: Optional[bool] = None) -> Array:
    """
    Element-wise logical :python:`x and y`.

    Calls :func:`jax.numpy.maximum` if approximation is enabled,
    :func:`jax.numpy.logical_or` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.multiply(x, y) if approx else jnp.logical_and(x, y)


@jit_approx
def logical_not(x: Array, *, approx: Optional[bool] = None) -> Array:
    """
    Element-wise logical :python:`not x`.

    Calls :func:`jax.numpy.subtract`
    (:python:`1 - x`) if approximation is enabled,
    :func:`jax.numpy.logical_or` otherwise.

    :param x: The input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
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
    """
    Element-wise logical :python:`x > y`.

    Calls :func:`jax.numpy.subtract`
    then :func:`sigmoid` if approximation is enabled,
    :func:`jax.numpy.greater` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return sigmoid(jnp.subtract(x, y), **kwargs) if approx else jnp.greater(x, y)


@jit_approx
def greater_equal(
    x: Array, y: Array, *, approx: Optional[bool] = None, **kwargs
) -> Array:
    """
    Element-wise logical :python:`x >= y`.

    Calls :func:`jax.numpy.subtract`
    then :func:`sigmoid` if approximation is enabled,
    :func:`jax.numpy.greater_equal` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return sigmoid(jnp.subtract(x, y), **kwargs) if approx else jnp.greater_equal(x, y)


@jit_approx
def less(x: Array, y: Array, *, approx: Optional[bool] = None, **kwargs) -> Array:
    """
    Element-wise logical :python:`x < y`.

    Calls :func:`jax.numpy.subtract` (arguments swapped)
    then :func:`sigmoid` if approximation is enabled,
    :func:`jax.numpy.less` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return sigmoid(jnp.subtract(y, x), **kwargs) if approx else jnp.less(x, y)


@jit_approx
def less_equal(x: Array, y: Array, *, approx: Optional[bool] = None, **kwargs) -> Array:
    """
    Element-wise logical :python:`x <= y`.

    Calls :func:`jax.numpy.subtract` (arguments swapped)
    then :func:`sigmoid` if approximation is enabled,
    :func:`jax.numpy.less_equal` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return sigmoid(jnp.subtract(y, x), **kwargs) if approx else jnp.less_equal(x, y)


@jit_approx
def is_true(x: Array, *, tol: float = 0.05, approx: Optional[bool] = None) -> Array:
    """
    Element-wise check if a given truth value can be considered to be true.

    When using approximation,
    this function checks whether the value is close to 1.

    :param x: The input array.
    :param tol: The tolerance on how close it should be to 1.
        Only used if :code:`approx` is set to :python:`True`.
    :param approx: Whether approximation is enabled or not.
    :return: True array if the value is considered to be true.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.greater(x, 1.0 - tol) if approx else jnp.asarray(x)


@jit_approx
def is_false(x: Array, *, tol: float = 0.05, approx: Optional[bool] = None) -> Array:
    """
    Element-wise check if a given truth value can be considered to be false.

    When using approximation,
    this function checks whether the value is close to 0.

    :param x: The input array.
    :param tol: The tolerance on how close it should be to 0.
        Only used if :code:`approx` is set to :python:`True`.
    :param approx: Whether approximation is enabled or not.
    :return: True if the value is considered to be false.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.less(x, tol) if approx else jnp.logical_not(x)
