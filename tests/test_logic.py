import chex
import jax
import jax.numpy as jnp
from jax import disable_jit

from differt2d.logic import enable_approx, is_true


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
