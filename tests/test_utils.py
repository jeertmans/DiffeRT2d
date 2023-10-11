from typing import List

import chex
import jax.numpy as jnp
import pytest
from chex import Array

from differt2d.geometry import RIS, Path, Wall
from differt2d.scene import Scene
from differt2d.utils import received_power, stack_leaves, unstack_leaves


def test_stack_and_unstack_leaves(key: Array):
    scene = Scene.random_uniform_scene(key, n_walls=10)
    walls = scene.objects

    assert all(isinstance(wall, Wall) for wall in walls)

    stacked_walls = stack_leaves(walls)

    assert isinstance(stacked_walls, Wall)

    unstacked_walls = unstack_leaves(stacked_walls)

    for w1, w2 in zip(walls, unstacked_walls):
        chex.assert_trees_all_equal(w1, w2)


def test_stack_and_unstack_different_pytrees(key: Array):
    scene = Scene.random_uniform_scene(key, n_walls=2)
    walls: List[Wall] = scene.objects  # type: ignore[assignment]
    walls[0] = RIS(points=walls[0].points)

    assert all(isinstance(wall, Wall) for wall in walls)

    with pytest.raises(ValueError):
        _ = stack_leaves(walls)


def test_received_power():
    path = Path(
        points=jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
    )

    expected = 0.3 / (2.0 * 2.0)
    got = received_power(None, None, path, None, r_coef=0.3, height=0.0)
    chex.assert_trees_all_close(got, expected)
