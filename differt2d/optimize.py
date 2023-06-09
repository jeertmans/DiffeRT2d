"""
Otimization toolbox.
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
    optimizer: optax.GradientTransformation = optax.adam(learning_rate=1),
) -> Tuple[X, Y]:
    """
    Minimizes a scalar function of one or more variables.

    :param fun: The objective function to be minimized.
    :param x0: The initial guess, (n,).
    :param steps: The number of steps to perform.
    :param optimizer: The optimizer to use.
    :return: The solution array and the corresponding loss.
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
    """
    keys = jax.random.split(key, num=many)

    @jax.jit
    def _minimize(key):
        return minimize_random_uniform(fun=fun, key=key, n=n, **kwargs)

    xs, losses = vmap(_minimize, in_axes=0)(keys)

    i_min = jnp.argmin(losses)

    return xs[i_min, :], losses[i_min]
