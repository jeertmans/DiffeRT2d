import chex
import jax
import jax.numpy as jnp
import pytest
from jax import disable_jit

from differt2d.logic import (
    activation,
    disable_approx,
    enable_approx,
    greater,
    greater_equal,
    is_false,
    is_true,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    true_value,
)

approx = pytest.mark.parametrize(("approx",), [(True,), (False,)])
alpha = pytest.mark.parametrize(
    ("alpha",), [(1e-3,), (1e-2,), (1e-1,), (1e-0,), (1e1,)]
)
function = pytest.mark.parametrize(("function",), [("sigmoid",), ("hard_sigmoid",)])
tol = pytest.mark.parametrize(("tol",), [(0.05,), (0.5,)])


@pytest.fixture
def x(key):
    x = jax.random.uniform(key, (200,))
    yield x


@pytest.fixture
def xy(key):
    key1, key2 = jax.random.split(key)
    x = jax.random.uniform(key1, (200,))
    y = jax.random.uniform(key2, (200,))
    yield x, y


def test_enable_approx():
    @jax.jit
    def approx_enabled():
        return jax.config.jax_enable_approx

    with enable_approx(True), disable_jit():
        assert jax.config.jax_enable_approx is True
        chex.assert_equal(True, approx_enabled())

    with enable_approx(False), disable_jit():
        assert jax.config.jax_enable_approx is False
        chex.assert_equal(False, approx_enabled())

    with enable_approx(True), disable_jit():
        expected = jnp.array(True)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(False)
        got = is_true(0.5)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(False)
        got = is_true(0.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    with enable_approx(False), disable_jit():
        expected = jnp.array(1.0)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(0.5)
        got = is_true(0.5)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(0.0)
        got = is_true(0.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(True)
        got = is_true(True)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(False)
        got = is_true(False)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


def test_enable_approx_clear_cache():
    is_true.clear_cache()
    with enable_approx(True):
        expected = jnp.array(True)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    is_true.clear_cache()
    with enable_approx(False):
        expected = jnp.array(1.0)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    is_true.clear_cache()
    with enable_approx(False):
        expected = jnp.array(True)
        got = is_true(True)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


def test_disable_approx():
    @jax.jit
    def approx_enabled():
        return jax.config.jax_enable_approx

    with disable_approx(False), disable_jit():
        assert jax.config.jax_enable_approx is True
        chex.assert_equal(True, approx_enabled())

    with disable_approx(True), disable_jit():
        assert jax.config.jax_enable_approx is False
        chex.assert_equal(False, approx_enabled())

    with disable_approx(False), disable_jit():
        expected = jnp.array(True)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(False)
        got = is_true(0.5)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(False)
        got = is_true(0.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    with disable_approx(True), disable_jit():
        expected = jnp.array(1.0)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(0.5)
        got = is_true(0.5)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(0.0)
        got = is_true(0.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(True)
        got = is_true(True)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)
        expected = jnp.array(False)
        got = is_true(False)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


def test_disable_approx_clear_cache():
    is_true.clear_cache()
    with disable_approx(False):
        expected = jnp.array(True)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    is_true.clear_cache()
    with disable_approx(True):
        expected = jnp.array(1.0)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    is_true.clear_cache()
    with disable_approx(True):
        expected = jnp.array(True)
        got = is_true(True)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


def test_enable_approx_with_keyword():
    expected = jnp.array(True)
    got = is_true(1.0, approx=True)
    chex.assert_trees_all_equal(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    expected = jnp.array(1.0)
    got = is_true(1.0, approx=False)
    chex.assert_trees_all_equal(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    expected = jnp.array(True)
    got = is_true(True, approx=False)
    chex.assert_trees_all_equal(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@pytest.mark.parametrize(
    ("function", "jax_fun"),
    [("sigmoid", jax.nn.sigmoid), ("hard_sigmoid", jax.nn.hard_sigmoid)],
)
@alpha
def test_activation(function, jax_fun, alpha):
    x = jnp.linspace(-5, +5, 200)
    expected = jax_fun(alpha * x)
    got = activation(x, alpha=alpha, function=function)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@pytest.mark.parametrize(
    ("function",), [("relu",), ("SIGMOID",), ("HARD_SIGMOID",), ("hard-sigmoid",)]
)
def test_invalid_activation(function):
    with pytest.raises(ValueError) as e:
        activation(1.0, function=function)
        assert "Unknown" in str(e)


@approx
def test_logical_or(xy, approx):
    x, y = xy
    if approx:
        expected = jnp.maximum(x, y)
    else:
        expected = jnp.logical_or(x, y)

    got = logical_or(x, y, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
def test_logical_and(xy, approx):
    x, y = xy
    if approx:
        expected = jnp.minimum(x, y)
    else:
        expected = jnp.logical_and(x, y)

    got = logical_and(x, y, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
def test_logical_not(x, approx):
    if approx:
        expected = jnp.subtract(1.0, x)
    else:
        expected = jnp.logical_not(x)

    got = logical_not(x, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@alpha
@function
def test_greater(xy, approx, alpha, function):
    x, y = xy
    if approx:
        expected = activation(x - y, alpha=alpha, function=function)
    else:
        expected = jnp.greater(x, y)

    got = greater(x, y, approx=approx, alpha=alpha, function=function)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@alpha
@function
def test_greater_equal(xy, approx, alpha, function):
    x, y = xy
    if approx:
        expected = activation(x - y, alpha=alpha, function=function)
    else:
        expected = jnp.greater_equal(x, y)

    got = greater_equal(x, y, approx=approx, alpha=alpha, function=function)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@alpha
@function
def test_less(xy, approx, alpha, function):
    x, y = xy
    if approx:
        expected = activation(y - x, alpha=alpha, function=function)
    else:
        expected = jnp.less(x, y)

    got = less(x, y, approx=approx, alpha=alpha, function=function)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@alpha
@function
def test_less_equal(xy, approx, alpha, function):
    x, y = xy
    if approx:
        expected = activation(y - x, alpha=alpha, function=function)
    else:
        expected = jnp.less_equal(x, y)

    got = less_equal(x, y, approx=approx, alpha=alpha, function=function)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@tol
def test_is_true(x, approx, tol):
    if approx:
        expected = jnp.greater(x, 1.0 - tol)
    else:
        x = jnp.greater(x, 1.0 - tol)
        expected = x

    got = is_true(x, tol=tol, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@tol
def test_is_false(x, approx, tol):
    if approx:
        expected = jnp.less(x, tol)
    else:
        x = jnp.greater(x, tol)
        expected = jnp.logical_not(x)

    got = is_false(x, tol=tol, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@tol
def test_true_value(approx, tol):
    x = true_value(approx=approx)
    assert is_true(x, tol=tol, approx=approx)
