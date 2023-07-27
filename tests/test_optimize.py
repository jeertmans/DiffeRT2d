import chex
import jax
import jax.numpy as jnp
import optax  # noqa: F401
import pytest
from jax import jit

from differt2d.optimize import (
    default_optimizer,
    minimize,
    minimize_many_random_uniform,
    minimize_random_uniform,
)


def convex_fun(x):
    x = x - 0.5
    return jnp.dot(x, x) + 2.0


def non_convex_fun(x):
    return jnp.dot(jnp.cos(x), jnp.sin(x))


jitted_convex_fun = jit(convex_fun)
jitted_non_convex_fun = jit(non_convex_fun)


@pytest.mark.parametrize(("fun",), [(convex_fun,), (jitted_convex_fun,)])
@pytest.mark.parametrize(("x0",), [([0.0],), ([1.0, 2.0, 3.0, 4.0],)])
def test_default_optimizer(fun, x0):
    x0 = jnp.atleast_1d(x0)
    expected_opt = default_optimizer()
    got_opt = eval(repr(expected_opt), globals())

    default_x, default_loss = minimize(fun, x0, steps=10)
    expected_x, expected_loss = minimize(fun, x0, steps=10, optimizer=expected_opt)
    got_x, got_loss = minimize(fun, x0, steps=10, optimizer=got_opt)

    chex.assert_trees_all_close(default_x, got_x)
    chex.assert_trees_all_equal_shapes_and_dtypes(default_x, got_x)
    chex.assert_trees_all_close(default_loss, got_loss)
    chex.assert_trees_all_equal_shapes_and_dtypes(default_loss, got_loss)

    chex.assert_trees_all_close(expected_x, got_x)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected_x, got_x)
    chex.assert_trees_all_close(expected_loss, got_loss)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected_loss, got_loss)


@pytest.mark.parametrize(("fun",), [(convex_fun,), (jitted_convex_fun,)])
@pytest.mark.parametrize(
    ("x0", "expected_x", "expected_loss"),
    [(0.0, 0.5, 2.0), (0.5, 0.5, 2.0), ([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], 2.0)],
)
def test_minimize(fun, x0, expected_x, expected_loss):
    x0 = jnp.atleast_1d(x0)
    expected_x = jnp.atleast_1d(expected_x)
    expected_loss = jnp.asarray(expected_loss)
    got_x, got_loss = minimize(fun, x0, steps=1000)
    chex.assert_trees_all_close(expected_x, got_x, rtol=1e-3)
    chex.assert_shape(got_x, expected_x.shape)
    chex.assert_trees_all_close(expected_loss, got_loss, rtol=1e-3)
    chex.assert_shape(got_loss, expected_loss.shape)


@pytest.mark.parametrize(("fun",), [(convex_fun,), (jitted_convex_fun,)])
@pytest.mark.parametrize(
    ("expected_x", "expected_loss"),
    [(0.5, 2.0), ([0.5, 0.5, 0.5], 2.0)],
)
def test_minimize_random_uniform(fun, seed, expected_x, expected_loss):
    expected_x = jnp.atleast_1d(expected_x)
    expected_loss = jnp.asarray(expected_loss)
    n = len(expected_x)
    key = jax.random.PRNGKey(seed)
    got_x, got_loss = minimize_random_uniform(fun, n=n, key=key, steps=1000)
    chex.assert_trees_all_close(expected_x, got_x, rtol=1e-3)
    chex.assert_shape(got_x, expected_x.shape)
    chex.assert_trees_all_close(expected_loss, got_loss, rtol=1e-3)
    chex.assert_shape(got_loss, expected_loss.shape)


@pytest.mark.parametrize(("fun",), [(convex_fun,), (jitted_convex_fun,)])
@pytest.mark.parametrize(
    ("expected_x", "expected_loss"),
    [(0.5, 2.0), ([0.5, 0.5, 0.5], 2.0)],
)
def test_minimize_many_random_uniform(fun, seed, expected_x, expected_loss):
    expected_x = jnp.atleast_1d(expected_x)
    expected_loss = jnp.asarray(expected_loss)
    n = len(expected_x)
    key = jax.random.PRNGKey(seed)
    got_x, got_loss = minimize_many_random_uniform(fun, n=n, key=key, steps=1000)
    chex.assert_trees_all_close(expected_x, got_x, rtol=1e-3)
    chex.assert_shape(got_x, expected_x.shape)
    chex.assert_trees_all_close(expected_loss, got_loss, rtol=1e-3)
    chex.assert_shape(got_loss, expected_loss.shape)
