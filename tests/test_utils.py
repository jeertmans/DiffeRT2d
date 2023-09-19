import chex
import jax
import pytest

from differt2d.geometry import RIS, Wall
from differt2d.scene import Scene
from differt2d.utils import flatten, stack_leaves, unstack_leaves


def test_stack_and_unstack_leaves(key: jax.random.PRNGKey):
    scene = Scene.random_uniform_scene(key, n_walls=10)
    walls = scene.objects

    assert all(isinstance(wall, Wall) for wall in walls)

    stacked_walls = stack_leaves(walls)

    assert isinstance(stacked_walls, Wall)

    unstacked_walls = unstack_leaves(stacked_walls)

    for w1, w2 in zip(walls, unstacked_walls):
        chex.assert_trees_all_equal(w1, w2)


def test_stack_and_unstack_different_pytrees(key: jax.random.PRNGKey):
    scene = Scene.random_uniform_scene(key, n_walls=2)
    walls = scene.objects
    walls[0] = RIS(points=walls[0].points)

    assert all(isinstance(wall, Wall) for wall in walls)

    with pytest.raises(ValueError):
        _ = stack_leaves(walls)


def test_flatten():
    nested = [1, [1, [1, 1, 1], 1], [[[1]]], [1], [[1]]]
    flattened = list(flatten(nested))

    assert len(flattened) == 9
    assert all(i == 1 for i in flattened)
