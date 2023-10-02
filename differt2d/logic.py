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
    "activation",
    "disable_approx",
    "enable_approx",
    "greater",
    "greater_equal",
    "is_false",
    "is_true",
    "less",
    "less_equal",
    "logical_all",
    "logical_and",
    "logical_any",
    "logical_not",
    "logical_or",
]

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .defaults import DEFAULT_ALPHA, DEFAULT_FUNCTION

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

_enable_approx = jax.config.define_bool_state(
    name="jax_enable_approx",
    default=True,
    help=("Enable approximation using some activation function."),
)

jit_approx = partial(jax.jit, inline=True, static_argnames=["approx", "function"])


@contextmanager
def enable_approx(enable: bool = True):
    """
    Context manager for enabling or disabling approximation of true/false values with
    continuous numbers from 0 (false) to 1 (true).

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
def disable_approx(disable: bool = True):  # pragma: no cover
    """
    Context manager for disable or enabling approximation of true/false values with
    continuous numbers from 0 (false) to 1 (true).

    This function is an alias to :python:`enable_approx(not disable)`.
    For more details, refer to :py:func:`enable_approx`.

    :param disable: Whether to disable or not approximation.

    .. note::

        Contrary to :py:func:`enable_approx`, there is no ``JAX_DISABLE_APPROX``
        environ variable, nor ``jax.config.jax_disable_approx`` config variable.
    """
    with _enable_approx(not disable):
        yield


@partial(jax.jit, inline=True, static_argnames=["function"])
def activation(
    x: Array,
    alpha: float = DEFAULT_ALPHA,
    function: Literal["sigmoid", "hard_sigmoid"] = DEFAULT_FUNCTION,
) -> Array:
    r"""
    Element-wise function for approximating a discrete transition between 0 and 1, with
    a smoothed transition centered at :python:`x = 0.0`.

    Depending on the ``function`` argument, the activation function has the
    following definition:

    .. math::
        \text{sigmoid}(x;\alpha) = \frac{1}{1 + e^{-\alpha x}},

    or

    .. math::
        \text{hard_sigmoid}(x;\alpha) = \frac{\text{relu6}(\alpha x+3)}{6},

    where :math:`\alpha` (:code:`alpha`) is a slope parameter.

    See :func:`jax.nn.sigmoid` or :func:`jax.nn.hard_sigmoid` for more details.

    :param x: The input array.
    :param alpha: The slope parameter.
    :return: The corresponding values.

    :EXAMPLES:

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import numpy as np
        from differt2d.logic import activation
        from jax import grad, vmap

        x = np.linspace(-5, +5, 200)

        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=[6.4, 8])

        for function in ["sigmoid", "hard_sigmoid"]:
            def f(x):
                return activation(x, alpha=1.0, function=function)

            y = f(x)
            dydx = vmap(grad(f))(x)
            _ = ax1.plot(x, y, "--", label=f"{function}")
            _ = ax2.plot(x, dydx, "-", label=f"{function}")

        ax2.set_xlabel("$x$")
        ax1.set_ylabel("$f(x)$")
        ax2.set_ylabel(r"$\frac{\partial f(x)}{\partial x}$")
        plt.legend()
        plt.tight_layout()
        plt.show()
    """
    if function == "sigmoid":
        return jax.nn.sigmoid(alpha * x)
    elif function == "hard_sigmoid":
        return jax.nn.hard_sigmoid(alpha * x)
    else:
        raise ValueError(f"Unknown function '{function}'")


@partial(jax.jit, inline=True, static_argnames=["approx"])
def logical_or(x: Array, y: Array, approx: Optional[bool] = None) -> Array:
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


@partial(jax.jit, inline=True, static_argnames=["approx"])
def logical_and(x: Array, y: Array, approx: Optional[bool] = None) -> Array:
    """
    Element-wise logical :python:`x and y`.

    Calls :func:`jax.numpy.minimum` if approximation is enabled,
    :func:`jax.numpy.logical_or` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.minimum(x, y) if approx else jnp.logical_and(x, y)


@partial(jax.jit, inline=True, static_argnames=["approx"])
def logical_not(x: Array, approx: Optional[bool] = None) -> Array:
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


@partial(jax.jit, inline=True, static_argnames=["approx", "function"])
def greater(
    x: Array,
    y: Array,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Array:
    """
    Element-wise logical :python:`x > y`.

    Calls :func:`jax.numpy.subtract`
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.greater` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments to be passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return activation(jnp.subtract(x, y), **kwargs) if approx else jnp.greater(x, y)


@partial(jax.jit, inline=True, static_argnames=["approx", "function"])
def greater_equal(
    x: Array, y: Array, approx: Optional[bool] = None, **kwargs: Any
) -> Array:
    """
    Element-wise logical :python:`x >= y`.

    Calls :func:`jax.numpy.subtract`
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.greater_equal` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments to be passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return (
        activation(jnp.subtract(x, y), **kwargs) if approx else jnp.greater_equal(x, y)
    )


@partial(jax.jit, inline=True, static_argnames=["approx", "function"])
def less(x: Array, y: Array, approx: Optional[bool] = None, **kwargs: Any) -> Array:
    """
    Element-wise logical :python:`x < y`.

    Calls :func:`jax.numpy.subtract` (arguments swapped)
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.less` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments to be passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return activation(jnp.subtract(y, x), **kwargs) if approx else jnp.less(x, y)


@partial(jax.jit, inline=True, static_argnames=["approx", "function"])
def less_equal(
    x: Array,
    y: Array,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Array:
    """
    Element-wise logical :python:`x <= y`.

    Calls :func:`jax.numpy.subtract` (arguments swapped)
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.less_equal` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments to be passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return activation(jnp.subtract(y, x), **kwargs) if approx else jnp.less_equal(x, y)


@partial(jax.jit, inline=True, static_argnames=["axis", "approx"])
def logical_all(
    *x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    approx: Optional[bool] = None,
) -> Array:
    """
    Returns whether all values in ``x`` are true.

    Calls :func:`jax.numpy.min` if approximation is enabled,
    :func:`jax.numpy.all` otherwise.

    :param x: The input array, or array-like.
    :param axis: Axis or axes along which to operate.
        By default, flattened input is used.
    :param approx: Whether approximation is enabled or not.
    :return: Output array.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    arr = jnp.asarray(x)
    return jnp.min(arr, axis=axis) if approx else jnp.all(arr, axis=axis)


@partial(jax.jit, inline=True, static_argnames=["axis", "approx"])
def logical_any(
    *x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    approx: Optional[bool] = None,
) -> Array:
    """
    Returns whether any value in ``x`` is true.

    Calls :func:`jax.numpy.max` if approximation is enabled,
    :func:`jax.numpy.any` otherwise.

    :param x: The input array, or array-like.
    :param axis: Axis or axes along which to operate.
        By default, flattened input is used.
    :param approx: Whether approximation is enabled or not.
    :return: Output array.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    arr = jnp.asarray(x)
    return jnp.max(arr, axis=axis) if approx else jnp.any(arr, axis=axis)


@partial(jax.jit, inline=True, static_argnames=["approx"])
def is_true(x: Array, tol: float = 0.5, approx: Optional[bool] = None) -> Array:
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


@partial(jax.jit, inline=True, static_argnames=["approx"])
def is_false(x: Array, tol: float = 0.5, approx: Optional[bool] = None) -> Array:
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


@partial(jax.jit, inline=True, static_argnames=["approx"])
def true_value(approx: Optional[bool] = None) -> Array:
    """
    Returns a scalar true value.

    When using approximation, this function returns 1.

    :param approx: Whether approximation is enabled or not.
    :return: A value that evaluates to true.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.array(1.0) if approx else jnp.array(True)


@partial(jax.jit, inline=True, static_argnames=["approx"])
def false_value(approx: Optional[bool] = None) -> Array:
    """
    Returns a scalar false value.

    When using approximation, this function returns 0.

    :param approx: Whether approximation is enabled or not.
    :return: A value that evaluates to false.
    """
    if approx is None:
        approx = jax.config.jax_enable_approx  # type: ignore[attr-defined]
    return jnp.array(0.0) if approx else jnp.array(False)
