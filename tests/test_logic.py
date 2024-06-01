import chex
import jax
import jax.numpy as jnp
import pytest
from jax import disable_jit
from jaxtyping import TypeCheckError

import differt2d.logic
from differt2d.logic import (
    activation,
    disable_approx,
    enable_approx,
    false_value,
    greater,
    greater_equal,
    hard_sigmoid,
    is_false,
    is_true,
    less,
    less_equal,
    logical_all,
    logical_and,
    logical_any,
    logical_not,
    logical_or,
    set_approx,
    sigmoid,
    true_value,
)

approx = pytest.mark.parametrize(("approx",), [(True,), (False,), (None,)])
alpha = pytest.mark.parametrize(
    ("alpha",), [(1e-3,), (1e-2,), (1e-1,), (1e-0,), (1e1,)]
)
function = pytest.mark.parametrize(("function",), [(sigmoid,), (hard_sigmoid,)])
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


def test_set_approx():
    set_approx(True)
    assert differt2d.logic.ENABLE_APPROX is True

    set_approx(False)
    assert differt2d.logic.ENABLE_APPROX is False


def test_enable_approx():
    @jax.jit
    def approx_enabled():
        return differt2d.logic.ENABLE_APPROX

    with enable_approx(True), disable_jit():
        assert differt2d.logic.ENABLE_APPROX is True
        chex.assert_equal(True, approx_enabled())

    with enable_approx(False), disable_jit():
        assert differt2d.logic.ENABLE_APPROX is False
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

    for value in [0.0, 0.5, 1.0]:
        with (
            enable_approx(False),
            disable_jit(),
            pytest.raises(TypeCheckError, match="Expected type: Bool"),
        ):
            got = is_true(value)

    for value in [True, False]:
        with enable_approx(False), disable_jit():
            expected = jnp.array(value)
            got = is_true(value)
            chex.assert_trees_all_equal(expected, got)
            chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


def test_enable_approx_clear_cache():
    is_true.clear_cache()  # type: ignore
    with enable_approx(True):
        expected = jnp.array(True)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    is_true.clear_cache()  # type: ignore
    with (
        enable_approx(False),
        pytest.raises(TypeCheckError, match="Expected type: Bool"),
    ):
        got = is_true(1.0)

    is_true.clear_cache()  # type: ignore
    with enable_approx(False):
        expected = jnp.array(True)
        got = is_true(True)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


def test_disable_approx():
    @jax.jit
    def approx_enabled():
        return differt2d.logic.ENABLE_APPROX

    with disable_approx(False), disable_jit():
        assert differt2d.logic.ENABLE_APPROX is True
        chex.assert_equal(True, approx_enabled())

    with disable_approx(True), disable_jit():
        assert differt2d.logic.ENABLE_APPROX is False
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

    for value in [0.0, 0.5, 1.0]:
        with (
            disable_approx(True),
            disable_jit(),
            pytest.raises(TypeCheckError, match="Expected type: Bool"),
        ):
            got = is_true(value)

    for value in [True, False]:
        with disable_approx(True), disable_jit():
            expected = jnp.array(value)
            got = is_true(value)
            chex.assert_trees_all_equal(expected, got)
            chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


def test_disable_approx_clear_cache():
    is_true.clear_cache()  # type: ignore
    with disable_approx(False):
        expected = jnp.array(True)
        got = is_true(1.0)
        chex.assert_trees_all_equal(expected, got)
        chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)

    is_true.clear_cache()  # type: ignore
    with (
        disable_approx(True),
        pytest.raises(TypeCheckError, match="Expected type: Bool"),
    ):
        got = is_true(1.0)

    is_true.clear_cache()  # type: ignore
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

    with pytest.raises(TypeCheckError, match="Expected type: Bool"):
        got = is_true(1.0, approx=False)

    expected = jnp.array(True)
    got = is_true(True, approx=False)
    chex.assert_trees_all_equal(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@pytest.mark.parametrize(
    ("function", "jax_fun"),
    [(sigmoid, jax.nn.sigmoid), (hard_sigmoid, jax.nn.hard_sigmoid)],
)
@alpha
def test_activation(function, jax_fun, alpha):
    x = jnp.linspace(-5, +5, 200)
    expected = jax_fun(alpha * x)
    got = activation(x, alpha=alpha, function=function)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
def test_logical_or(xy, approx):
    x, y = xy
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
        expected = jnp.maximum(x, y)
    else:
        expected = jnp.logical_or(x, y)

    got = logical_or(x, y, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
def test_logical_and(xy, approx):
    x, y = xy
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
        expected = jnp.minimum(x, y)
    else:
        expected = jnp.logical_and(x, y)

    got = logical_and(x, y, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
def test_logical_not(x, approx):
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
        expected = jnp.subtract(1.0, x)
    else:
        expected = jnp.logical_not(x)

    got = logical_not(x, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
def test_logical_all(x, approx):
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
        x = jnp.array([0.8, 0.2, 0.3])
        expected = jnp.min(x)
    else:
        x = jnp.array([True, False, False])
        expected = jnp.all(x)

    got = logical_all(*x, axis=0, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
def test_logical_any(x, approx):
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
        x = jnp.array([0.8, 0.2, 0.3])
        expected = jnp.max(x)
    else:
        x = jnp.array([True, False, False])
        expected = jnp.any(x)

    got = logical_any(*x, axis=0, approx=approx)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@alpha
@function
def test_greater(xy, approx, alpha, function):
    x, y = xy
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
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
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
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
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
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
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
        expected = activation(y - x, alpha=alpha, function=function)
    else:
        expected = jnp.less_equal(x, y)

    got = less_equal(x, y, approx=approx, alpha=alpha, function=function)
    chex.assert_trees_all_close(expected, got)
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, got)


@approx
@tol
def test_is_true(x, approx, tol):
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
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
    if approx or (approx is None and differt2d.logic.ENABLE_APPROX):
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


@approx
@tol
def test_false_value(approx, tol):
    x = false_value(approx=approx)
    assert is_false(x, tol=tol, approx=approx)
