"""
Otimization toolbox.

All the functions present in this toolbox support
Just-in-time compilation with :func:`jax.jit`.

Examples
========

>>> from differt2d.optimize import minimize
>>> import chex
>>> import jax
>>> import jax.numpy as jnp
>>> @jax.jit
... def parabola_min(a, b, c):
...     def f(x):
...         x = a * (x + b) + c
...         return jnp.dot(x, x)
...
...     return minimize(f, jnp.array(0.0))
>>>
>>> x, y = parabola_min(2.0, 1.0, 1.0)
>>> chex.assert_trees_all_close(x, -1.5, rtol=1e-2)
>>> chex.assert_trees_all_close(y, +0.0, atol=1e-3)
"""

from typing import Any, Callable, Tuple, TypeVar

import jax
import jax.numpy as jnp
import optax
from jax import Array, vmap

X = TypeVar("X", bound=Array)
Y = TypeVar("Y", bound=Array)


def minimize(
    fun: Callable[[X], Y],
    x0: Array,
    steps: int = 100,
    optimizer: optax.GradientTransformation = optax.adam(learning_rate=0.1),
) -> Tuple[X, Y]:
    """
    Minimizes a scalar function of one or more variables.

    :param fun: The objective function to be minimized.
    :param x0: The initial guess, (n,).
    :param steps: The number of steps to perform.
    :param optimizer: The optimizer to use,
        defaults to :func:`optax.adam` with :python:`learning_rate = .1`.
    :return: The solution array and the corresponding loss.

    :Examples:

    >>> from differt2d.optimize import minimize
    >>> import chex
    >>> import jax.numpy as jnp
    >>> def f(x):
    ...     x = x - 1.0
    ...     return jnp.dot(x, x)
    >>>
    >>> x, y = minimize(f, jnp.zeros(10))
    >>> chex.assert_trees_all_close(x, jnp.ones(10), rtol=1e-2)
    >>> chex.assert_trees_all_close(y, 0.0, atol=1e-4)
    """
    f_and_df = jax.value_and_grad(fun)
    opt_state = optimizer.init(x0)

    @jax.jit
    def f(carry, x):
        x, opt_state = carry
        loss, grads = f_and_df(x)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = x + updates
        carry = (x, opt_state)
        return carry, loss

    (x, _), losses = jax.lax.scan(f, init=(x0, opt_state), xs=None, length=steps)
    return x, losses[-1]


def minimize_random_uniform(
    fun: Callable[[X], Y],
    key: jax.random.KeyArray,
    n: int,
    **kwargs: Any,
) -> Tuple[X, Y]:
    """
    Minimizes a scalar function of one or more variables,
    with initial guess drawn randomly from a uniform distribution.

    :param fun: The objective function to be minimized.
    :param key: The random key to generate the initial guess.
    :param n: The size of the random vector to generate.
    :param kwargs:
        Keyword arguments to be passed to :func:`minimize`.
    :return: The solution array and the corresponding loss.

    :Examples:

    >>> from differt2d.optimize import minimize_random_uniform
    >>> import chex
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x):
    ...     x = x - 1.0
    ...     return jnp.dot(x, x)
    >>>
    >>> x, y = minimize_random_uniform(f, jax.random.PRNGKey(1234), 10)
    >>> chex.assert_trees_all_close(x, jnp.ones(10), rtol=1e-2)
    >>> chex.assert_trees_all_close(y, 0.0, atol=1e-3)
    """
    x0 = jax.random.uniform(key, shape=(n,))
    return minimize(fun=fun, x0=x0, **kwargs)


def minimize_many_random_uniform(
    fun: Callable[[X], Y],
    key: jax.random.KeyArray,
    n: int,
    many: int = 10,
    **kwargs: Any,
) -> Tuple[X, Y]:
    """
    Minimizes many times a scalar function of one or more variables,
    with initial guess drawn randomly from a uniform distribution,
    and returns the best minimum out of the :code:`many` trials.

    :param fun: The objective function to be minimized.
    :param key: The random key to generate the initial guess.
    :param n: The size of the random vector to generate.
    :param many:
        How many times the minimization should be performed.
    :param kwargs:
        Keyword arguments to be passed to :func:`minimize`.
    :return: The solution array and the corresponding loss.

    :Examples:

    >>> from differt2d.optimize import minimize_many_random_uniform
    >>> import chex
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x):
    ...     x = x - 1.0
    ...     return jnp.dot(x, x)
    >>>
    >>> x, y = minimize_many_random_uniform(f, jax.random.PRNGKey(1234), 10)
    >>> chex.assert_trees_all_close(x, jnp.ones(10), rtol=1e-2)
    >>> chex.assert_trees_all_close(y, 0.0, atol=1e-4)
    """
    keys = jax.random.split(key, num=many)

    @jax.jit
    def _minimize(key):
        return minimize_random_uniform(fun=fun, key=key, n=n, **kwargs)

    xs, losses = vmap(_minimize, in_axes=0)(keys)

    i_min = jnp.argmin(losses)

    return xs[i_min, :], losses[i_min]
