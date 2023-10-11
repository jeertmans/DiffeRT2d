from json.decoder import JSONDecodeError
from pathlib import Path
from typing import get_args

import chex
import jax.numpy as jnp
import pytest

from differt2d.geometry import Point
from differt2d.scene import Scene, SceneName


@pytest.fixture
def geojson_file():
    file = Path(__file__).parent / "example.geojson"
    assert file.exists()
    yield file


class TestScene:
    @pytest.mark.parametrize(
        ("transform",),
        [
            (lambda p: p.read_text(),),
            (lambda p: p.read_bytes(),),
            (lambda p: bytearray(p.read_bytes()),),
            (lambda p: p.open(),),
        ],
    )
    def test_from_geojson(self, geojson_file, transform):
        s_or_fp = transform(geojson_file)
        scene = Scene.from_geojson(s_or_fp, tx_loc="SW", rx_loc="NE")
        bounding_box = scene.bounding_box()
        assert len(scene.transmitters) == 1
        assert len(scene.objects) == 28
        assert len(scene.receivers) == 1
        chex.assert_trees_all_equal(scene.transmitters["tx"].point, bounding_box[0, :])
        chex.assert_trees_all_equal(scene.receivers["rx"].point, bounding_box[1, :])

    def test_from_geojson_unimplemented(self, geojson_file):
        with pytest.raises(NotImplementedError):
            _ = Scene.from_geojson(geojson_file)

    def test_from_geojson_decode_error(self, geojson_file):
        with pytest.raises(JSONDecodeError):
            _ = Scene.from_geojson(geojson_file.as_posix())

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
        scene = Scene.random_uniform_scene(key, n_transmitters=n)

        assert isinstance(scene, Scene)
        assert len(scene.transmitters) == n
        assert len(scene.objects) == 1
        assert len(scene.receivers) == 1

        scene = Scene.random_uniform_scene(key, n_walls=n)

        assert isinstance(scene, Scene)
        assert len(scene.transmitters) == 1
        assert len(scene.objects) == n
        assert len(scene.receivers) == 1

        scene = Scene.random_uniform_scene(key, n_receivers=n)

        assert isinstance(scene, Scene)
        assert len(scene.transmitters) == 1
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
        scene = Scene.random_uniform_scene(key, n_transmitters=3, n_walls=5, n_receivers=2)
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
            [scene.transmitters["tx_0"].point, scene.receivers["rx_0"].point]
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

    def test_get_closest_transmitter(self, key):
        scene = Scene.random_uniform_scene(key, n_transmitters=10)
        expected_point = Point(point=jnp.array([0.5, 0.5]))
        expected_distance = jnp.array(0.0)
        scene.transmitters["closest"] = expected_point
        got_point, got_distance = scene.get_closest_transmitter(expected_point.point)
        chex.assert_trees_all_equal(got_point, expected_point)
        chex.assert_trees_all_equal(got_distance, expected_distance)

    def test_get_closest_receiver(self, key):
        scene = Scene.random_uniform_scene(key, n_receivers=10)
        expected_point = Point(point=jnp.array([0.5, 0.5]))
        expected_distance = jnp.array(0.0)
        scene.receivers["closest"] = expected_point
        got_point, got_distance = scene.get_closest_receiver(expected_point.point)
        chex.assert_trees_all_equal(got_point, expected_point)
        chex.assert_trees_all_equal(got_distance, expected_distance)

    def test_accumulate_over_paths(self):
        pass

    def test_accumulate_on_transmitters_grid_over_paths(self):
        def fun(transmitter, receiver, path, interacting_objects):
            return path.length() ** 2  # = x^2 + y^2 in LOS

        rx0 = Point(point=jnp.array([0.0, 0.0]))
        rx1 = Point(point=jnp.array([0.0, 1.0]))
        scene = Scene(transmitters={}, objects=[], receivers=dict(rx0=rx0, rx1=rx1))

        x = y = jnp.linspace(-3, 3, 10)
        X, Y = jnp.meshgrid(x, y)
        got_Z = scene.accumulate_on_transmitters_grid_over_paths(
            X, Y, fun=fun, max_order=1, approx=False
        )

        rx_key, got_Z0 = next(got_Z)
        assert rx_key == "rx0"

        rx_key, got_Z1 = next(got_Z)
        assert rx_key == "rx1"

        chex.assert_trees_all_equal_shapes_and_dtypes(X, got_Z0)
        chex.assert_trees_all_equal_shapes_and_dtypes(Y, got_Z1)

        expected_Z0 = X**2 + Y**2
        chex.assert_trees_all_close(got_Z0, expected_Z0)

        expected_Z1 = X**2 + (Y - 1.0) ** 2
        chex.assert_trees_all_close(got_Z1, expected_Z1)

        got_Z = scene.accumulate_on_transmitters_grid_over_paths(
            X, Y, fun=fun, reduce_all=True, max_order=1, approx=False
        )
        expected_Z = expected_Z0 + expected_Z1
        chex.assert_trees_all_close(got_Z, expected_Z)

        got_dZ = scene.accumulate_on_transmitters_grid_over_paths(
            X, Y, fun=fun, reduce_all=True, grad=True, max_order=1, approx=False
        )
        expected_dZ0 = jnp.stack([2 * X, 2 * Y], axis=-1)
        expected_dZ1 = jnp.stack([2 * X, 2 * (Y - 1.0)], axis=-1)
        expected_dZ = expected_dZ0 + expected_dZ1
        chex.assert_trees_all_close(got_dZ, expected_dZ)

        got_Z, got_dZ = scene.accumulate_on_transmitters_grid_over_paths(
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
        def fun(transmitter, receiver, path, interacting_objects):
            return path.length() ** 2  # = x^2 + y^2 in LOS

        tx0 = Point(point=jnp.array([0.0, 0.0]))
        tx1 = Point(point=jnp.array([1.0, 0.0]))
        scene = Scene(transmitters=dict(tx0=tx0, tx1=tx1), objects=[], receivers={})

        x = y = jnp.linspace(-3, 3, 10)
        X, Y = jnp.meshgrid(x, y)
        got_Z = scene.accumulate_on_receivers_grid_over_paths(
            X, Y, fun=fun, max_order=1, approx=False
        )

        tx_key, got_Z0 = next(got_Z)
        assert tx_key == "tx0"

        tx_key, got_Z1 = next(got_Z)
        assert tx_key == "tx1"

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
