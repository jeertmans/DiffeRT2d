import chex
import jax.numpy as jnp
import pytest

from differt2d.geometry import Wall


class TestWall:
    @pytest.fixture
    def wall(self) -> Wall:
        return Wall(path=jnp.array([[0.0, 0.0], [1.0, 2.0]]))

    def test_invalid_wall(self):
        with pytest.raises(AssertionError):
            _ = Wall(path=jnp.array([[0.0, 0.0]]))

    def test_origin(self, wall: Wall):
        expected = wall.path[0]
        got = wall.origin
        chex.assert_trees_all_equal(expected, got)
