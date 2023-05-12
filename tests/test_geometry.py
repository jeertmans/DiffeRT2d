import chex
import jax.numpy as jnp
import pytest

from differt2d.geometry import Ray, Wall

origin_dest = pytest.mark.parametrize(
    ("origin", "dest"),
    [
        ([0.0, 0.0], [0.0, 1.0]),  # Vertical
        ([0.0, 0.0], [1.0, 0.0]),  # Horizontal
        ([0.0, 0.0], [1.0, 1.0]),  # Diagonal
        ([1.0, 1.0], [0.0, 0.0]),  # Dianonal, reversed
    ],
)


class TestRay:
    @pytest.mark.parametrize(
        "path",
        [
            0.0,  # Scalar
            [0.0, 0.0],
            [[0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
    )
    def test_invalid_path(self, path: list):
        with pytest.raises(AssertionError):
            _ = Ray(path=jnp.array(path))

    @origin_dest
    def test_origin(self, origin: list, dest: list):
        expected = jnp.array(origin)
        got = Ray(path=jnp.array([origin, dest])).origin()
        chex.assert_trees_all_equal(expected, got)

    @origin_dest
    def test_dest(self, origin: list, dest: list):
        expected = jnp.array(dest)
        got = Ray(path=jnp.array([origin, dest])).dest()
        chex.assert_trees_all_equal(expected, got)


class TestWall:
    @origin_dest
    def test_normal(self, origin: list, dest: list):
        v = jnp.array(dest) - jnp.array(origin)
        w = Wall(path=jnp.array([origin, dest]))
        normal = w.normal_at_t(0.0)
        chex.assert_trees_all_close(jnp.dot(v, normal), 0.0, atol=1e-7)
        chex.assert_trees_all_close(jnp.linalg.norm(normal), 1.0)

        for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
            other_normal = w.normal_at_t(t)
            chex.assert_trees_all_equal(normal, other_normal)
