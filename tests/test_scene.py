from typing import get_args

import chex
import jax.numpy as jnp
import pytest

from differt2d.geometry import Point
from differt2d.scene import Scene, SceneName


class TestScene:
    @pytest.mark.parametrize(
        ("scene_name",),
        [(scene_name,) for scene_name in get_args(SceneName)],
    )
    def test_from_scene_name(self, scene_name):
        _ = Scene.from_scene_name(scene_name)

    @pytest.mark.parametrize(
        ("n",),
        [
            (1,),
            (2,),
            (3,),
            (4,),
        ],
    )
    def test_random_uniform_scene(self, key, n):
        scene = Scene.random_uniform_scene(key, n_emitters=n)

        assert isinstance(scene, Scene)
        assert len(scene.emitters) == n
        assert len(scene.objects) == 1
        assert len(scene.receivers) == 1

        scene = Scene.random_uniform_scene(key, n_walls=n)

        assert isinstance(scene, Scene)
        assert len(scene.emitters) == 1
        assert len(scene.objects) == n
        assert len(scene.receivers) == 1

        scene = Scene.random_uniform_scene(key, n_receivers=n)

        assert isinstance(scene, Scene)
        assert len(scene.emitters) == 1
        assert len(scene.objects) == 1
        assert len(scene.receivers) == n

    def test_basic_scene(self):
        scene = Scene.basic_scene()

        assert isinstance(scene, Scene)

    def test_square_scene(self):
        scene = Scene.square_scene()

        assert isinstance(scene, Scene)
        assert len(scene.objects) == 4

    def test_square_scene_with_obstacle(self):
        scene = Scene.square_scene_with_obstacle()

        assert isinstance(scene, Scene)
        assert len(scene.objects) == 8

    def test_square_scene_with_wall(self):
        scene = Scene.square_scene_with_obstacle()

        assert isinstance(scene, Scene)
        assert len(scene.objects) == 8

    def test_plot(self, ax, key):
        scene = Scene.random_uniform_scene(key, n_emitters=3, n_walls=5, n_receivers=2)
        _ = scene.plot(ax)

        scene = Scene.basic_scene()
        _ = scene.plot(ax)

        scene = Scene.square_scene()
        _ = scene.plot(ax)

        scene = Scene.square_scene_with_obstacle()
        _ = scene.plot(ax)

    def test_bounding_box(self, key):
        scene = Scene.random_uniform_scene(key, n_walls=10)

        points = jnp.vstack(
            [scene.emitters["tx_0"].point, scene.receivers["rx_0"].point]
            + [obj.points for obj in scene.objects]
        )

        expected = jnp.array(
            [
                [jnp.min(points[:, 0]), jnp.min(points[:, 1])],
                [jnp.max(points[:, 0]), jnp.max(points[:, 1])],
            ]
        )
        got = scene.bounding_box()
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, (2, 2))

    def test_accumulate_on_emitters_grid_over_paths(self):
        def fun(emitter, receiver, path, interacting_objects):
            return path.length() ** 2  # = x^2 + y^2 in LOS

        rx0 = Point(point=jnp.array([0.0, 0.0]))
        rx1 = Point(point=jnp.array([0.0, 1.0]))
        scene = Scene(emitters={}, objects=[], receivers=dict(rx0=rx0, rx1=rx1))

        x = y = jnp.linspace(-3, 3, 10)
        X, Y = jnp.meshgrid(x, y)
        got_Z = scene.accumulate_on_emitters_grid_over_paths(
            X, Y, fun=fun, max_order=1, approx=False
        )

        e_key, got_Z0 = next(got_Z)
        assert e_key == "rx0"

        e_key, got_Z1 = next(got_Z)
        assert e_key == "rx1"

        chex.assert_trees_all_equal_shapes_and_dtypes(X, got_Z0)
        chex.assert_trees_all_equal_shapes_and_dtypes(Y, got_Z1)

        expected_Z0 = X**2 + Y**2
        chex.assert_trees_all_close(got_Z0, expected_Z0)

        expected_Z1 = X**2 + (Y - 1.0) ** 2
        chex.assert_trees_all_close(got_Z1, expected_Z1)

        got_Z = scene.accumulate_on_emitters_grid_over_paths(
            X, Y, fun=fun, reduce_all=True, max_order=1, approx=False
        )
        expected_Z = expected_Z0 + expected_Z1
        chex.assert_trees_all_close(got_Z, expected_Z)

        got_dZ = scene.accumulate_on_emitters_grid_over_paths(
            X, Y, fun=fun, reduce_all=True, grad=True, max_order=1, approx=False
        )
        expected_dZ0 = jnp.stack([2 * X, 2 * Y], axis=-1)
        expected_dZ1 = jnp.stack([2 * X, 2 * (Y - 1.0)], axis=-1)
        expected_dZ = expected_dZ0 + expected_dZ1
        chex.assert_trees_all_close(got_dZ, expected_dZ)

        got_Z, got_dZ = scene.accumulate_on_emitters_grid_over_paths(
            X,
            Y,
            fun=fun,
            reduce_all=True,
            value_and_grad=True,
            max_order=1,
            approx=False,
        )
        chex.assert_trees_all_close(got_Z, expected_Z)
        chex.assert_trees_all_close(got_dZ, expected_dZ)

    def test_accumulate_on_receivers_grid_over_paths(self):
        def fun(emitter, receiver, path, interacting_objects):
            return path.length() ** 2  # = x^2 + y^2 in LOS

        tx0 = Point(point=jnp.array([0.0, 0.0]))
        tx1 = Point(point=jnp.array([1.0, 0.0]))
        scene = Scene(emitters=dict(tx0=tx0, tx1=tx1), objects=[], receivers={})

        x = y = jnp.linspace(-3, 3, 10)
        X, Y = jnp.meshgrid(x, y)
        got_Z = scene.accumulate_on_receivers_grid_over_paths(
            X, Y, fun=fun, max_order=1, approx=False
        )

        e_key, got_Z0 = next(got_Z)
        assert e_key == "tx0"

        e_key, got_Z1 = next(got_Z)
        assert e_key == "tx1"

        chex.assert_trees_all_equal_shapes_and_dtypes(X, got_Z0)
        chex.assert_trees_all_equal_shapes_and_dtypes(Y, got_Z1)

        expected_Z0 = X**2 + Y**2
        chex.assert_trees_all_close(got_Z0, expected_Z0)

        expected_Z1 = (X - 1.0) ** 2 + Y**2
        chex.assert_trees_all_close(got_Z1, expected_Z1)

        got_Z = scene.accumulate_on_receivers_grid_over_paths(
            X, Y, fun=fun, reduce_all=True, max_order=1, approx=False
        )
        expected_Z = expected_Z0 + expected_Z1
        chex.assert_trees_all_close(got_Z, expected_Z)

        got_dZ = scene.accumulate_on_receivers_grid_over_paths(
            X, Y, fun=fun, reduce_all=True, grad=True, max_order=1, approx=False
        )
        expected_dZ0 = jnp.stack([2 * X, 2 * Y], axis=-1)
        expected_dZ1 = jnp.stack([2 * (X - 1.0), 2 * Y], axis=-1)
        expected_dZ = expected_dZ0 + expected_dZ1
        chex.assert_trees_all_close(got_dZ, expected_dZ)

        got_Z, got_dZ = scene.accumulate_on_receivers_grid_over_paths(
            X,
            Y,
            fun=fun,
            reduce_all=True,
            value_and_grad=True,
            max_order=1,
            approx=False,
        )
        chex.assert_trees_all_close(got_Z, expected_Z)
        chex.assert_trees_all_close(got_dZ, expected_dZ)
