import chex
import jax
import pytest

from differt2d.geometry import RIS, Wall
from differt2d.scene import Scene
from differt2d.utils import flatten, stack_leaves, unstack_leaves, patch
from differt2d.logic import true_value


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



def add(a: int, b: int = 0):
    return a + b


def add_plus_one(*args, **kwargs):
    return add(*args, **kwargs) + 1


def test_patch():
    with patch(add, b=1):

        assert add(1) == 2
        assert add(1, 2) == 2
        assert add(1, b=3) == 2

        assert add_plus_one(1) == 3
        assert add_plus_one(1, 2) == 3
        assert add_plus_one(1, b=3) == 3

    with patch(add, a=4):

        assert add(1) == 4
        assert add(1, 2) == 6
        assert add(1, b=3) == 7

        assert add_plus_one(1) == 5
        assert add_plus_one(1, 2) == 7
        assert add_plus_one(1, b=3) == 8

    assert add(1) == 1
    assert add(1, 2) == 3
    assert add(1, b=3) == 4

    assert add_plus_one(1) == 2
    assert add_plus_one(1, 2) == 4
    assert add_plus_one(1, b=3) == 5
