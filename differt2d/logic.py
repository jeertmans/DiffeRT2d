"""
A toolbox for logical operations.

When approximation is enabled, a value close to 1 maps to :py:data:`True`,
while a value close to 0 maps :py:data:`False`.

Otherwise, functions will call their JAX counterpart. E.g.,
:py:func:`logical_or` calls :py:func:`jax.numpy.logical_or`
when :code:`approx` is set to :py:data:`False`.

.. note::

    Whenever a function takes an argument named ``approx``, it can take
    three different values:

    1. :py:data:`None`: defaults to :py:data:`differt2d.logic.ENABLE_APPROX`,
       see :py:func:`enable_approx` for comments on that;
    2. :py:data:`True`: forces to enable approximation;
    3. or :py:data:`False`: forces to disable approximation.
"""

__all__ = (
    "ENABLE_APPROX",
    "Truthy",
    "activation",
    "set_approx",
    "disable_approx",
    "enable_approx",
    "greater",
    "greater_equal",
    "hard_sigmoid",
    "is_false",
    "is_true",
    "less",
    "less_equal",
    "logical_all",
    "logical_and",
    "logical_any",
    "logical_not",
    "logical_or",
    "sigmoid",
)

import os
from contextlib import contextmanager
from functools import partial
from threading import Lock
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, jaxtyped

from ._typing import ArrayLikeFloat, ScalarFloat
from .defaults import DEFAULT_ALPHA

ENABLE_APPROX: bool = "ENABLE_APPROX" in os.environ
"""Enable approximation using some activation function."""

_LOCK = Lock()
"""Lock to prevent mutating ``ENABLE_APPROX`` in multiple threads."""

Truthy = Union[Bool[Array, " *batch"], Float[Array, " *batch"]]
"""An array of truthy values, either booleans or floats between 0 and 1."""


def set_approx(enable: bool):
    """
    Enable or disable the approximation in future function calls.

    Note that JIT-compiled version will not be affected if they
    were compiled before this function was called.

    :param enable: Whether to enable or not approximation.

    :Examples:

    >>> from differt2d.logic import greater, set_approx
    >>>
    >>> # doc: hide
    >>> greater.clear_cache()
    >>> # doc: hide
    >>> set_approx(False)
    >>> print(greater(20.0, 5.0))
    True
    """
    global ENABLE_APPROX

    ENABLE_APPROX = enable


@contextmanager
def enable_approx(enable: bool = True):
    """
    Context manager for enabling or disabling approximation of true/false values with continuous numbers from 0 (false) to 1 (true).

    By default, approximation is enabled.

    To disable approximation, you have multiple options:

    1. use this context manager to disable it (see example below);
    2. set the environ variable ``DISABLE_APPROX`` (any value);
    3. update the config with
       :py:func:`set_approx(False)`;
    4. or set, for specific logic functions only, the keyword argument
       ``approx`` to :py:data:`False`.

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

        Calling already-jitted functions after mutating ``ENABLE_APPROX``
        will not produce any visible change. This is because
        :py:data:`ENABLE_APPROX` is evaluated once, at compilation.

        For example:

        >>> import jax
        >>> import differt2d.logic
        >>> from differt2d.logic import enable_approx
        >>>
        >>> @jax.jit
        ... def f():
        ...     if differt2d.logic.ENABLE_APPROX:
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
        ...     if differt2d.logic.ENABLE_APPROX:
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
    global ENABLE_APPROX
    state = ENABLE_APPROX
    with _LOCK:
        try:
            ENABLE_APPROX = enable
            yield
        finally:
            ENABLE_APPROX = state


@contextmanager
def disable_approx(disable: bool = True):  # pragma: no cover
    """
    Context manager for disable or enabling approximation of true/false values with continuous numbers from 0 (false) to 1 (true).

    This function is an alias to :python:`enable_approx(not disable)`.
    For more details, refer to :py:func:`enable_approx`.

    :param disable: Whether to disable or not approximation.

    .. note::

        Contrary to :py:func:`enable_approx`, there is no ``DISABLE_APPROX``
        environ variable, nor ``differt2d.logic.DISABLE_APPROX`` config variable.
    """
    with enable_approx(not disable):
        yield


@partial(jax.jit, inline=True)
@jaxtyped(typechecker=typechecker)
def sigmoid(x: ArrayLikeFloat, alpha: ScalarFloat) -> Float[Array, " *batch"]:
    r"""
    Element-wise sigmoid, parametrized with ``alpha``.

    .. math::
        \text{sigmoid}(x;\alpha) = \frac{1}{1 + e^{-\alpha x}},

    where :math:`\alpha` (``alpha``) is a slope parameter.

    See :func:`jax.nn.sigmoid` for more details.

    :param x: The input array.
    :param alpha: The slope parameter.
    :return: The corresponding values.
    """
    return jax.nn.sigmoid(alpha * x)


@partial(jax.jit, inline=True)
@jaxtyped(typechecker=typechecker)
def hard_sigmoid(x: ArrayLikeFloat, alpha: ScalarFloat) -> Float[Array, " *batch"]:
    r"""
    Element-wise sigmoid, parametrized with ``alpha``.

    .. math::
        \text{hard_sigmoid}(x;\alpha) = \frac{\text{relu6}(\alpha x + 3)}{6},

    where :math:`\alpha` (``alpha``) is a slope parameter.

    See :func:`jax.nn.hard_sigmoid` and :func:`jax.nn.relu6` for more details.

    :param x: The input array.
    :param alpha: The slope parameter.
    :return: The corresponding values.
    """
    return jax.nn.hard_sigmoid(alpha * x)


@partial(jax.jit, inline=True, static_argnames=("function",))
@jaxtyped(typechecker=None)
def activation(
    x: ArrayLikeFloat,
    alpha: ScalarFloat = DEFAULT_ALPHA,
    function: Callable[
        [ArrayLikeFloat, ScalarFloat],
        Float[Array, " *batch"],
    ] = hard_sigmoid,
) -> Float[Array, " *batch"]:
    r"""
    Element-wise function for approximating a discrete transition between 0 and 1, with a smoothed transition centered at :python:`x = 0.0`.

    Depending on the ``function`` argument, the activation function has the
    different definition.

    Two basic activation functions are provided: :func:`sigmoid` and :func:`hard_sigmoid`.
    If needed, you can implement your own activation function and pass it as an argument,
    granted that it satisfies the properties defined in the related paper.

    :param x: The input array.
    :param alpha: The slope parameter.
    :return: The corresponding values.

    :EXAMPLES:

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import numpy as np
        from differt2d.logic import activation, hard_sigmoid, sigmoid
        from jax import grad, vmap

        x = np.linspace(-5, +5, 200)

        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=[6.4, 8])

        for name, function in [("sigmoid", sigmoid), ("hard_sigmoid", hard_sigmoid)]:
            def f(x):
                return activation(x, alpha=1.5, function=function)

            y = f(x)
            dydx = vmap(grad(f))(x)
            _ = ax1.plot(x, y, "--", label=f"{name}")
            _ = ax2.plot(x, dydx, "-", label=f"{name}")

        ax2.set_xlabel("$x$")
        ax1.set_ylabel("$f(x)$")
        ax2.set_ylabel(r"$\frac{\partial f(x)}{\partial x}$")
        plt.legend()
        plt.tight_layout()
        plt.show()  # doctest: +SKIP
    """
    return function(x, alpha)


@partial(jax.jit, inline=True, static_argnames=("approx",))
@jaxtyped(typechecker=typechecker)
def logical_or(
    x: Union[Truthy, float, bool],
    y: Union[Truthy, float, bool],
    approx: Optional[bool] = None,
) -> Truthy:
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
        approx = ENABLE_APPROX
    return jnp.maximum(x, y) if approx else jnp.logical_or(x, y)


@partial(jax.jit, inline=True, static_argnames=("approx",))
@jaxtyped(typechecker=typechecker)
def logical_and(
    x: Union[Truthy, float, bool],
    y: Union[Truthy, float, bool],
    approx: Optional[bool] = None,
) -> Truthy:
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
        approx = ENABLE_APPROX
    return jnp.minimum(x, y) if approx else jnp.logical_and(x, y)


@partial(jax.jit, inline=True, static_argnames=("approx",))
@jaxtyped(typechecker=typechecker)
def logical_not(x: Union[Truthy, float, bool], approx: Optional[bool] = None) -> Truthy:
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
        approx = ENABLE_APPROX
    return jnp.subtract(1.0, x) if approx else jnp.logical_not(x)


@partial(jax.jit, inline=True, static_argnames=("approx", "function"))
@jaxtyped(typechecker=typechecker)
def greater(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    """
    Element-wise logical :python:`x > y`.

    Calls :func:`jax.numpy.subtract`
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.greater` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = ENABLE_APPROX
    return activation(jnp.subtract(x, y), **kwargs) if approx else jnp.greater(x, y)


@partial(jax.jit, inline=True, static_argnames=("approx", "function"))
@jaxtyped(typechecker=typechecker)
def greater_equal(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    """
    Element-wise logical :python:`x >= y`.

    Calls :func:`jax.numpy.subtract`
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.greater_equal` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = ENABLE_APPROX
    return (
        activation(jnp.subtract(x, y), **kwargs) if approx else jnp.greater_equal(x, y)
    )


@partial(jax.jit, inline=True, static_argnames=("approx", "function"))
@jaxtyped(typechecker=typechecker)
def less(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    """
    Element-wise logical :python:`x < y`.

    Calls :func:`jax.numpy.subtract` (arguments swapped)
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.less` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = ENABLE_APPROX
    return activation(jnp.subtract(y, x), **kwargs) if approx else jnp.less(x, y)


@partial(jax.jit, inline=True, static_argnames=("approx", "function"))
@jaxtyped(typechecker=typechecker)
def less_equal(
    x: ArrayLikeFloat,
    y: ArrayLikeFloat,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    """
    Element-wise logical :python:`x <= y`.

    Calls :func:`jax.numpy.subtract` (arguments swapped)
    then :func:`activation` if approximation is enabled,
    :func:`jax.numpy.less_equal` otherwise.

    :param x: The first input array.
    :param y: The second input array.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation`.
    :return: Output array, with element-wise comparison.
    """
    if approx is None:
        approx = ENABLE_APPROX
    return activation(jnp.subtract(y, x), **kwargs) if approx else jnp.less_equal(x, y)


@partial(jax.jit, inline=True, static_argnames=("axis", "approx"))
@jaxtyped(typechecker=typechecker)
def logical_all(
    *x: Union[Truthy, float, bool],
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    approx: Optional[bool] = None,
) -> Truthy:
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
        approx = ENABLE_APPROX
    arr = jnp.asarray(x)
    return jnp.min(arr, axis=axis) if approx else jnp.all(arr, axis=axis)


@partial(jax.jit, inline=True, static_argnames=("axis", "approx"))
@jaxtyped(typechecker=typechecker)
def logical_any(
    *x: Union[Truthy, float, bool],
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    approx: Optional[bool] = None,
) -> Truthy:
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
        approx = ENABLE_APPROX
    arr = jnp.asarray(x)
    return jnp.max(arr, axis=axis) if approx else jnp.any(arr, axis=axis)


@partial(jax.jit, inline=True, static_argnames=("approx",))
@jaxtyped(typechecker=typechecker)
def is_true(
    x: Union[Truthy, float, bool],
    tol: ScalarFloat = 0.5,
    approx: Optional[bool] = None,
) -> Bool[Array, " *batch"]:
    """
    Element-wise check if a given truth value can be considered to be true.

    When using approximation,
    this function checks whether the value is close to 1.

    :param x: The input array.
    :param tol: The tolerance on how close it should be to 1.
        Only used if :code:`approx` is set to :py:data:`True`.
    :param approx: Whether approximation is enabled or not.
    :return: True array if the value is considered to be true.
    """
    if approx is None:
        approx = ENABLE_APPROX
    return jnp.greater(x, 1.0 - tol) if approx else jnp.asarray(x)


@partial(jax.jit, inline=True, static_argnames=("approx",))
@jaxtyped(typechecker=typechecker)
def is_false(
    x: Union[Truthy, float, bool],
    tol: ScalarFloat = 0.5,
    approx: Optional[bool] = None,
) -> Bool[Array, " *batch"]:
    """
    Element-wise check if a given truth value can be considered to be false.

    When using approximation,
    this function checks whether the value is close to 0.

    :param x: The input array.
    :param tol: The tolerance on how close it should be to 0.
        Only used if :code:`approx` is set to :py:data:`True`.
    :param approx: Whether approximation is enabled or not.
    :return: True if the value is considered to be false.
    """
    if approx is None:
        approx = ENABLE_APPROX
    return jnp.less(x, tol) if approx else jnp.logical_not(x)


@partial(jax.jit, inline=False, static_argnames=("approx",))
@jaxtyped(typechecker=typechecker)
def true_value(approx: Optional[bool] = None) -> Truthy:
    """
    Returns a scalar true value.

    When using approximation, this function returns 1.

    :param approx: Whether approximation is enabled or not.
    :return: A value that evaluates to true.
    """
    if approx is None:
        approx = ENABLE_APPROX
    return jnp.array(1.0) if approx else jnp.array(True, dtype=bool)


@partial(jax.jit, inline=False, static_argnames=("approx",))
@jaxtyped(typechecker=typechecker)
def false_value(approx: Optional[bool] = None) -> Truthy:
    """
    Returns a scalar false value.

    When using approximation, this function returns 0.

    :param approx: Whether approximation is enabled or not.
    :return: A value that evaluates to false.
    """
    if approx is None:
        approx = ENABLE_APPROX
    return jnp.array(0.0) if approx else jnp.array(False, dtype=bool)
