from json.decoder import JSONDecodeError
from pathlib import Path
from typing import get_args

import chex
import jax.numpy as jnp
import pytest

from differt2d.geometry import RIS, Point, Wall, stack_leaves
from differt2d.logic import is_true
from differt2d.scene import PyTreeDict, Scene, SceneName


@pytest.fixture
def geojson_file():
    file = Path(__file__).parent / "example.geojson"
    assert file.exists()
    yield file


class TestPyTreeDict:
    def test_check_init(self):
        with pytest.raises(
            ValueError, match="Number of keys must match number of values"
        ):
            _ = PyTreeDict(_keys=("a", "b"), _values=("key_a",))

    def test_from_mapping(self):
        m = {"a": 1, "b": 2}
        d = PyTreeDict.from_mapping(m)

        for a, b in zip(m, d):
            assert a == b

    def test_get_items(self):
        m = {"a": 1, "b": 2}
        d = PyTreeDict.from_mapping(m)

        assert d["a"] == 1
        assert d["b"] == 2

        with pytest.raises(KeyError):
            _ = d["c"]

    def test_contains(self):
        m = {"a": 1, "b": 2}
        d = PyTreeDict.from_mapping(m)

        assert "a" in d
        assert "b" in d
        assert "c" not in d


class TestScene:
    def test_empty_scene(self):
        scene = Scene()

        assert len(scene.transmitters) == 0
        assert len(scene.receivers) == 0
        assert len(scene.objects) == 0

    def test_with_transmitters(self):
        scene = Scene(transmitters={"a": Point(), "b": Point()})

        assert len(scene.transmitters) == 2

        scene = scene.with_transmitters(c=Point())

        assert len(scene.transmitters) == 1

    def test_with_receivers(self):
        scene = Scene(receivers={"a": Point(), "b": Point()})

        assert len(scene.receivers) == 2

        scene = scene.with_receivers(c=Point())

        assert len(scene.receivers) == 1

    def test_with_objects(self):
        scene = Scene(objects=[Wall(), Wall()])

        assert len(scene.objects) == 2

        scene = scene.with_objects(Wall())

        assert len(scene.objects) == 1

    def test_filter_objects(self):
        scene = Scene(objects=[Wall(), Wall()])
        ris = RIS()
        scene = scene.add_objects(ris)

        assert len(scene.objects) == 3

        scene = scene.filter_objects(lambda o: isinstance(o, RIS))

        assert len(scene.objects) == 1
        assert isinstance(scene.objects[0], RIS)

    def test_update_transmitters(self):
        scene = Scene(transmitters={"a": Point(), "b": Point()})

        assert len(scene.transmitters) == 2

        scene = scene.update_transmitters(c=Point())

        assert len(scene.transmitters) == 3

    def test_update_receivers(self):
        scene = Scene(receivers={"a": Point(), "b": Point()})

        assert len(scene.receivers) == 2

        scene = scene.update_receivers(c=Point())

        assert len(scene.receivers) == 3

    def test_add_objects(self):
        scene = Scene(objects=[Wall(), Wall()])

        assert len(scene.objects) == 2

        scene = scene.add_objects(Wall())

        assert len(scene.objects) == 3

    def test_rename_transmitters(self):
        scene = Scene(transmitters={"a": Point(), "b": Point()})

        assert "a" in scene.transmitters

        scene = scene.rename_transmitters(a="A")

        assert "a" not in scene.transmitters
        assert "A" in scene.transmitters
        assert len(scene.transmitters) == 2

        scene = scene.rename_transmitters(c="C")

        assert "c" not in scene.transmitters
        assert "C" not in scene.transmitters
        assert len(scene.transmitters) == 2

    def test_rename_receivers(self):
        scene = Scene(receivers={"a": Point(), "b": Point()})

        assert "a" in scene.receivers

        scene = scene.rename_receivers(a="A")

        assert "a" not in scene.receivers
        assert "A" in scene.receivers
        assert len(scene.receivers) == 2

        scene = scene.rename_receivers(c="C")

        assert "c" not in scene.receivers
        assert "C" not in scene.receivers
        assert len(scene.receivers) == 2

    def test_from_walls_array(self, key):
        scene = Scene.random_uniform_scene(
            n_transmitters=0, n_receivers=0, n_walls=30, key=key
        )

        walls = scene.objects
        walls = stack_leaves(walls)

        walls_array = walls.xys

        assert walls_array.shape == (30, 2, 2)

        new_scene = Scene.from_walls_array(walls_array)

        chex.assert_trees_all_equal(scene, new_scene)

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

        if hasattr(s_or_fp, "close"):
            s_or_fp.close()  # Important to close file

        bounding_box = scene.bounding_box()
        assert len(scene.transmitters) == 1
        assert len(scene.objects) == 28
        assert len(scene.receivers) == 1
        chex.assert_trees_all_equal(scene.transmitters["tx"].xy, bounding_box[0, :])
        chex.assert_trees_all_equal(scene.receivers["rx"].xy, bounding_box[1, :])

    def test_from_empty_geojson(self):
        scene = Scene.from_geojson('{"features": []}')

        assert len(scene.transmitters) == 1
        assert len(scene.receivers) == 1
        assert len(scene.objects) == 0

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
        scene = Scene.random_uniform_scene(key=key, n_transmitters=n)

        assert isinstance(scene, Scene)
        assert len(scene.transmitters) == n
        assert len(scene.objects) == 1
        assert len(scene.receivers) == 1

        scene = Scene.random_uniform_scene(key=key, n_walls=n)

        assert isinstance(scene, Scene)
        assert len(scene.transmitters) == 1
        assert len(scene.objects) == n
        assert len(scene.receivers) == 1

        scene = Scene.random_uniform_scene(key=key, n_receivers=n)

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
        scene = Scene.random_uniform_scene(
            key=key, n_transmitters=3, n_walls=5, n_receivers=2
        )
        _ = scene.plot(ax)

        scene = Scene.basic_scene()
        _ = scene.plot(ax)

        scene = Scene.square_scene()
        _ = scene.plot(ax)

        scene = Scene.square_scene_with_obstacle()
        _ = scene.plot(ax)

    def test_bounding_box(self, key):
        scene = Scene.random_uniform_scene(key=key, n_walls=10)

        points = jnp.vstack(
            [scene.transmitters["tx_0"].xy, scene.receivers["rx_0"].xy]
            + [obj.xys for obj in scene.objects]
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
        scene = Scene.random_uniform_scene(key=key, n_transmitters=10)
        expected_point = Point(xy=jnp.array([0.5, 0.5]))
        expected_distance = jnp.array(0.0)
        scene = scene.update_transmitters(closest=expected_point)
        got_point_name, got_distance = scene.get_closest_transmitter(expected_point.xy)
        got_point = scene.transmitters[got_point_name]
        chex.assert_trees_all_equal(got_point, expected_point)
        chex.assert_trees_all_equal(got_distance, expected_distance)

    def test_get_closest_receiver(self, key):
        scene = Scene.random_uniform_scene(key=key, n_receivers=10)
        expected_point = Point(xy=jnp.array([0.5, 0.5]))
        expected_distance = jnp.array(0.0)
        scene = scene.update_receivers(closest=expected_point)
        got_point_name, got_distance = scene.get_closest_receiver(expected_point.xy)
        got_point = scene.receivers[got_point_name]
        chex.assert_trees_all_equal(got_point, expected_point)
        chex.assert_trees_all_equal(got_distance, expected_distance)

    def test_all_path_candidates(self, key):
        scene = Scene.random_uniform_scene(key=key, n_receivers=10)
        got = scene.all_path_candidates(min_order=0, max_order=0)
        assert len(got) == 1
        assert len(got[0]) == 0
        got = scene.all_path_candidates(order=0)
        assert len(got) == 1
        assert len(got[0]) == 0

    def test_all_path_candidates_filter_objects(self):
        scene = Scene(objects=[Wall(), Wall(), Wall()])
        ris = RIS()
        scene = scene.add_objects(ris, Wall(), Wall())

        assert len(scene.objects) == 6

        # Index of RIS is 3
        expected = [
            jnp.empty(0, dtype=jnp.int32),
            jnp.array([3], dtype=jnp.int32),
        ]
        got = scene.all_path_candidates(
            filter_objects=lambda o: isinstance(o, RIS), min_order=0, max_order=2
        )

        chex.assert_trees_all_equal(got, expected)

    @pytest.mark.parametrize(
        ("min_order", "max_order"), [(0, 0), (1, 1), (2, 2), (0, 2)]
    )
    def test_all_paths_and_valid_paths(self, min_order, max_order, key):
        scene = Scene.square_scene()

        valid_paths = scene.all_valid_paths(
            approx=False,
            min_order=min_order,
            max_order=max_order,
            key=key,  # Key is unused here
        )

        for tx_key, rx_key, got_valid, expected_path, path_candidate in scene.all_paths(
            min_order=min_order,
            max_order=max_order,
            approx=False,
        ):
            assert tx_key == "tx"
            assert rx_key == "rx"

            assert min_order <= expected_path.xys.shape[0] - 2 <= max_order
            assert min_order <= len(path_candidate) <= max_order

            interacting_objects = scene.get_interacting_objects(path_candidate)
            expected_valid = expected_path.is_valid(
                scene.objects, path_candidate, interacting_objects, approx=False
            )

            chex.assert_trees_all_equal(got_valid, expected_valid)

            if is_true(got_valid, approx=False):
                _, _, got_path, _ = next(valid_paths)
                chex.assert_trees_all_equal(got_path, expected_path)

        with pytest.raises(StopIteration):
            _ = next(valid_paths)

    def test_accumulate_over_paths(self):
        def fun(transmitter, receiver, path, interacting_objects):
            return path.length() ** 2  # = x^2 + y^2 in LOS

        tx0 = Point(xy=jnp.array([0.0, 0.0]))
        tx1 = Point(xy=jnp.array([1.0, 0.0]))
        rx0 = Point(xy=jnp.array([1.0, 1.0]))
        rx1 = Point(xy=jnp.array([0.0, 1.0]))
        scene = Scene(
            transmitters=dict(tx0=tx0, tx1=tx1),
            objects=[],
            receivers=dict(rx0=rx0, rx1=rx1),
        )

        got = scene.accumulate_over_paths(fun=fun, max_order=1, approx=False)

        tx_key, rx_key, acc = next(got)  # type: ignore
        assert tx_key == "tx0"
        assert rx_key == "rx0"
        chex.assert_trees_all_close(acc, jnp.array(2.0))

        tx_key, rx_key, acc = next(got)  # type: ignore
        assert tx_key == "tx0"
        assert rx_key == "rx1"
        chex.assert_trees_all_close(acc, jnp.array(1.0))

        tx_key, rx_key, acc = next(got)  # type: ignore
        assert tx_key == "tx1"
        assert rx_key == "rx0"
        chex.assert_trees_all_close(acc, jnp.array(1.0))

        tx_key, rx_key, acc = next(got)  # type: ignore
        assert tx_key == "tx1"
        assert rx_key == "rx1"
        chex.assert_trees_all_close(acc, jnp.array(2.0))

        got = scene.accumulate_over_paths(
            fun=fun, reduce_all=True, max_order=1, approx=False
        )
        chex.assert_trees_all_close(got, jnp.array(6.0))

    def test_accumulate_on_transmitters_grid_over_paths(self, key):
        def fun(transmitter, receiver, path, interacting_objects):
            return path.length() ** 2  # = x^2 + y^2 in LOS

        rx0 = Point(xy=jnp.array([0.0, 0.0]))
        rx1 = Point(xy=jnp.array([0.0, 1.0]))
        scene = Scene(transmitters={}, objects=[], receivers=dict(rx0=rx0, rx1=rx1))

        x = y = jnp.linspace(-3, 3, 10)
        X, Y = jnp.meshgrid(x, y)
        got_Z = scene.accumulate_on_transmitters_grid_over_paths(
            X, Y, fun=fun, max_order=1, approx=False, key=key
        )

        rx_key, got_Z0 = next(got_Z)  # type: ignore
        assert rx_key == "rx0"

        rx_key, got_Z1 = next(got_Z)  # type: ignore
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

    def test_accumulate_on_receivers_grid_over_paths(self, key):
        def fun(transmitter, receiver, path, interacting_objects):
            return path.length() ** 2  # = x^2 + y^2 in LOS

        tx0 = Point(xy=jnp.array([0.0, 0.0]))
        tx1 = Point(xy=jnp.array([1.0, 0.0]))
        scene = Scene(transmitters=dict(tx0=tx0, tx1=tx1), objects=[], receivers={})

        x = y = jnp.linspace(-3, 3, 10)
        X, Y = jnp.meshgrid(x, y)
        got_Z = scene.accumulate_on_receivers_grid_over_paths(
            X, Y, fun=fun, max_order=1, approx=False, key=key
        )

        tx_key, got_Z0 = next(got_Z)  # type: ignore
        assert tx_key == "tx0"

        tx_key, got_Z1 = next(got_Z)  # type: ignore
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
