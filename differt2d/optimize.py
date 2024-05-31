"""
Optimization toolbox.

All the functions present in this toolbox support
Just-in-time compilation with :func:`jax.jit`.

Examples
--------
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

import sys
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:  # pragma: no cover
    from typing_extensions import TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")


@partial(jax.jit, static_argnames=("fun", "steps", "optimizer"))
@jaxtyped(typechecker=None)
def minimize(
    fun: Callable[[Float[Array, " n"], *Ts], Float[Array, " "]],
    x0: Float[Array, " n"],
    args: tuple[Unpack[Ts]] = (),
    steps: int = 100,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> tuple[Float[Array, " n"], Float[Array, " "]]:
    """
    Minimizes a scalar function of one or more variables.

    :param fun: The objective function to be minimized.
    :param x0: The initial guess.
    :param args:
        Positional arguments passed to ``fun``.
    :param steps: The number of steps to perform.
    :param optimizer: The optimizer to use. If not provided,
        uses :func:`optax.adam` with a learning rate of ``0.1``.
    :return: The solution array and the corresponding loss.

    :Examples:

    >>> from differt2d.optimize import minimize
    >>> import chex
    >>> import jax.numpy as jnp
    >>> def f(x, offset=1.0):
    ...     x = x - offset
    ...     return jnp.dot(x, x)
    >>>
    >>> x, y = minimize(f, jnp.zeros(10))
    >>> chex.assert_trees_all_close(x, jnp.ones(10), rtol=1e-2)
    >>> chex.assert_trees_all_close(y, 0.0, atol=1e-4)
    >>>
    >>> # It is also possible to pass positional arguments
    >>> x, y = minimize(f, jnp.zeros(10), args=(2.0,))
    >>> chex.assert_trees_all_close(x, 2.0 * jnp.ones(10), rtol=1e-2)
    >>> chex.assert_trees_all_close(y, 0.0, atol=1e-3)
    """
    optimizer = optimizer if optimizer else optax.adam(learning_rate=0.1)

    f_and_df = jax.value_and_grad(fun)
    opt_state = optimizer.init(x0)

    def f(carry, x):
        x, opt_state = carry
        loss, grads = f_and_df(x, *args)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = x + updates
        carry = (x, opt_state)
        return carry, loss

    (x, _), losses = jax.lax.scan(f, init=(x0, opt_state), xs=None, length=steps)
    return x, losses[-1]


@partial(jax.jit, static_argnames=("fun", "n", "steps", "optimizer"))
@jaxtyped(typechecker=None)
def minimize_random_uniform(
    fun: Callable[[Float[Array, " {n}"], *Ts], Float[Array, " "]],
    key: PRNGKeyArray,
    n: int,
    **kwargs: Any,
) -> tuple[Float[Array, " {n}"], Float[Array, " "]]:
    """
    Minimizes a scalar function of one or more variables, with initial guess drawn randomly from a uniform distribution.

    :param fun: The objective function to be minimized.
    :param key: The random key used to generate the initial guess.
    :param n: The size of the random vector to generate.
    :param kwargs:
        Keyword arguments passed to :func:`minimize`.
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


@partial(jax.jit, static_argnames=("fun", "n", "many", "steps", "optimizer"))
@jaxtyped(typechecker=None)
def minimize_many_random_uniform(
    fun: Callable[[Float[Array, " {n}"], *Ts], Float[Array, " "]],
    key: PRNGKeyArray,
    n: int,
    many: int = 10,
    **kwargs: Any,
) -> tuple[Float[Array, " {n}"], Float[Array, " "]]:
    """
    Minimizes many times a scalar function of one or more variables, with initial guess drawn randomly from a uniform distribution, and returns the best minimum out of the :code:`many` trials.

    :param fun: The objective function to be minimized.
    :param key: The random key used to generate the initial guesses.
    :param n: The size of the random vector to generate.
    :param many:
        How many times the minimization should be performed.
    :param kwargs:
        Keyword arguments passed to :func:`minimize_random_uniform`.
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
    if many == 1:
        return minimize_random_uniform(fun=fun, key=key, n=n, **kwargs)

    keys = jax.random.split(key, num=many)

    def _minimize(key):
        return minimize_random_uniform(fun=fun, key=key, n=n, **kwargs)

    xs, losses = jax.vmap(_minimize, in_axes=0)(keys)

    i_min = jnp.argmin(losses)

    return xs[i_min, :], losses[i_min]
