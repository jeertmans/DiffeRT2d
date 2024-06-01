import chex
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from differt2d.geometry import Wall


class TestPlottable:
    def test_grid(self):
        wall = Wall(xys=jnp.array([[0.0, 0.0], [1.0, 2.0]]))

        X, Y = wall.grid(25)

        assert X.shape == (25, 25)
        assert Y.shape == (25, 25)
        assert float(X.min()) == 0.0
        assert float(X.max()) == 1.0
        assert float(Y.min()) == 0.0
        assert float(Y.max()) == 2.0

    def test_center(self):
        wall = Wall(xys=jnp.array([[0.0, 1.0], [1.0, 2.0]]))

        got = wall.center()
        expected = jnp.array([0.5, 1.5])

        chex.assert_trees_all_equal(got, expected)

    def test_get_location(self):
        wall = Wall(xys=jnp.array([[0.0, 0.5], [1.0, 2.0]]))

        got = wall.get_location("N")
        expected = jnp.array([0.5, 2.0])

        chex.assert_trees_all_equal(got, expected)

        got = wall.get_location("NW")
        expected = jnp.array([0.0, 2.0])

        chex.assert_trees_all_equal(got, expected)

        got = wall.get_location("NE")
        expected = jnp.array([1.0, 2.0])

        chex.assert_trees_all_equal(got, expected)

        got = wall.get_location("W")
        expected = jnp.array([0.0, 1.25])

        chex.assert_trees_all_equal(got, expected)

        got = wall.get_location("E")
        expected = jnp.array([1.0, 1.25])

        chex.assert_trees_all_equal(got, expected)

        got = wall.get_location("SE")
        expected = jnp.array([1.0, 0.5])

        chex.assert_trees_all_equal(got, expected)

        got = wall.get_location("S")
        expected = jnp.array([0.5, 0.5])

        chex.assert_trees_all_equal(got, expected)

        got = wall.get_location("SW")
        expected = jnp.array([0.0, 0.5])

        got = wall.get_location("C")
        expected = wall.center()

        chex.assert_trees_all_equal(got, expected)

        with pytest.raises(TypeCheckError):
            _ = wall.get_location("L")  # type: ignore
