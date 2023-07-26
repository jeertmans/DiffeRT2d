import chex
import jax
import jax.numpy as jnp
import pytest
from jax import disable_jit

from differt2d.logic import activation, enable_approx, is_true


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
@pytest.mark.parametrize(("alpha",), [(1e-3,), (1e-2,), (1e-1,), (1e-0,), (1e1,)])
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
