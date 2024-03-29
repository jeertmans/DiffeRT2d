from typing import Type

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import Array
from jax import disable_jit

from differt2d.geometry import (
    RIS,
    FermatPath,
    ImagePath,
    MinPath,
    Path,
    Point,
    Ray,
    Wall,
    path_length,
    segments_intersect,
)
from differt2d.logic import enable_approx, false_value, is_false, is_true, true_value
from differt2d.scene import Scene

origin_dest = pytest.mark.parametrize(
    ("origin", "dest"),
    [
        ([0.0, 0.0], [0.0, 1.0]),  # Vertical
        ([0.0, 0.0], [1.0, 0.0]),  # Horizontal
        ([0.0, 0.0], [1.0, 1.0]),  # Diagonal
        ([1.0, 1.0], [0.0, 0.0]),  # Dianonal, reversed
        ([0.81891436, 0.97124588], [0.78140455, 0.59715754]),  # Random 1
        ([0.73011344, 0.04696848], [0.34776926, 0.50277104]),  # Random 2
    ],
)

point = pytest.mark.parametrize(
    ("point",),
    [
        ([0.0, 0.0],),
        ([1.0, 0.0],),
        ([0.0, 1.0],),
        ([1.0, 1.0],),
        ([0.61653844, 0.72739276],),
        ([0.34096069, 0.62302206],),
    ],
)

approx = pytest.mark.parametrize(
    ("approx",),
    [
        (True,),
        (False,),
    ],
)

path_cls = pytest.mark.parametrize(
    ("path_cls",),
    [
        (Path,),
        (ImagePath,),
        (FermatPath,),
        (MinPath,),
    ],
)


@pytest.fixture
def steps():
    return 100


@approx
def test_segments_intersect(approx: bool):
    P1 = jnp.array([+0.0, +0.0])
    P2 = jnp.array([+1.0, +0.0])
    P3 = jnp.array([+0.5, -1.0])
    P4 = jnp.array([+0.5, +1.0])
    intersect = segments_intersect(P1, P2, P3, P4, approx=approx)

    assert is_true(intersect, approx=approx)


@approx
def test_segments_dont_intersect(approx: bool):
    P1 = jnp.array([+0.0, +0.0])
    P2 = jnp.array([+1.0, +0.0])
    P3 = jnp.array([+0.0, +1.0])
    P4 = jnp.array([+1.0, +1.0])
    intersect = segments_intersect(P1, P2, P3, P4, approx=approx)

    assert is_false(intersect, approx=approx)


def test_path_length():
    points = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    expected = jnp.array(4.0)
    got = path_length(points)
    chex.assert_trees_all_equal(expected, got)


class TestRay:
    @origin_dest
    def test_origin(self, origin: list, dest: list):
        expected = jnp.array(origin)
        got = Ray(points=jnp.array([origin, dest])).origin()
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, (2,))

    @origin_dest
    def test_dest(self, origin: list, dest: list):
        expected = jnp.array(dest)
        got = Ray(points=jnp.array([origin, dest])).dest()
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, (2,))

    @origin_dest
    def test_t(self, origin: list, dest: list):
        expected = jnp.array(dest) - jnp.array(origin)
        got = Ray(points=jnp.array([origin, dest])).t()
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, (2,))

    @origin_dest
    def test_plot(self, ax, origin: list, dest: list):
        _ = Ray(points=jnp.array([origin, dest])).plot(ax)

    @origin_dest
    def test_bounding_box(self, origin: list, dest: list):
        points = jnp.array([origin, dest])
        expected = jnp.array(
            [
                [jnp.min(points[:, 0]), jnp.min(points[:, 1])],
                [jnp.max(points[:, 0]), jnp.max(points[:, 1])],
            ]
        )
        got = Ray(points=points).bounding_box()
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, (2, 2))


class TestPoint:
    @point
    def test_plot(self, ax, point: list):
        _ = Point(point=jnp.array(point)).plot(ax)

    @point
    def test_bounding_box(self, point: list):
        point = jnp.array(point)
        expected = jnp.array(
            [
                [point[0], point[1]],
                [point[0], point[1]],
            ]
        )
        got = Point(point=point).bounding_box()
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, (2, 2))


class TestWall:
    @origin_dest
    def test_normal(self, origin: list, dest: list):
        v = jnp.array(dest) - jnp.array(origin)
        w = Wall(points=jnp.array([origin, dest]))
        normal = w.normal()
        chex.assert_trees_all_close(jnp.dot(v, normal), 0.0, atol=1e-7)
        chex.assert_trees_all_close(jnp.linalg.norm(normal), 1.0)

    def test_parameters_count(self):
        got = Wall.parameters_count()
        chex.assert_trees_all_equal(got, 1)
        chex.assert_shape(got, ())

    def test_parametric_to_cartesian(self):
        expected = jnp.array([2.0, 1.0])
        got = Wall(points=jnp.array([[0.0, 0.0], [4.0, 2.0]])).parametric_to_cartesian(
            0.5
        )
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, (2,))

    def test_cartesian_to_parametric(self):
        wall = Wall(points=jnp.array([[0.0, 0.0], [4.0, 2.0]]))
        expected = jnp.array(0.5)
        got = wall.cartesian_to_parametric(jnp.array([2.0, 1.0]))
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, ())
        expected = jnp.array(0.0)
        got = wall.cartesian_to_parametric(jnp.array([0.0, 0.0]))
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, ())
        expected = jnp.array(1.0)
        got = wall.cartesian_to_parametric(jnp.array([4.0, 2.0]))
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, ())
        expected = jnp.array(2.0)
        got = wall.cartesian_to_parametric(jnp.array([8.0, 4.0]))
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, ())
        expected = jnp.array(-1.0)
        got = wall.cartesian_to_parametric(jnp.array([-4.0, -2.0]))
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, ())

    @approx
    def test_contains_parametric(self, approx: bool):
        wall = Wall(points=jnp.array([[0.0, 0.0], [4.0, 2.0]]))
        with enable_approx(approx), disable_jit():
            got = wall.contains_parametric(0.5)
            assert is_true(got)
            chex.assert_shape(got, ())
            got = wall.contains_parametric(2.0)
            assert is_false(got)
            chex.assert_shape(got, ())

    @approx
    def test_intersects_cartesian(self, approx: bool):
        wall = Wall(points=jnp.array([[0.0, 0.0], [4.0, 2.0]]))
        with enable_approx(approx), disable_jit():
            got = wall.intersects_cartesian(jnp.array([[0.0, 2.0], [4.0, 0.0]]))
            assert is_true(got)
            chex.assert_shape(got, ())
            got = wall.intersects_cartesian(jnp.array([[0.0, 1.0], [4.0, 3.0]]))
            assert is_false(got)
            chex.assert_shape(got, ())
            got = wall.intersects_cartesian(jnp.array([[0.0, 1.0], [2.0, 7.0]]))
            assert is_false(got)
            chex.assert_shape(got, ())
            got = wall.intersects_cartesian(jnp.array([[0.0, 1.0], [0.0, 0.0]]))
            if approx:
                assert got > 0, "Should intersect even on the extremity"
            else:
                assert is_true(got), "Should intersect even on the extremity"
            chex.assert_shape(got, ())

    def test_evaluate_cartesian(self):
        wall = Wall(points=jnp.array([[0.0, 0.0], [4.0, 0.0]]))
        expected = jnp.array(0.0)
        ray_path = jnp.array([[0.0, 1.0], [2.0, 0.0], [4.0, 1.0]])
        got = wall.evaluate_cartesian(ray_path)
        chex.assert_trees_all_close(expected, got)
        chex.assert_shape(got, ())
        ray_path = jnp.array([[0.0, 1.0], [2.1, 0.0], [4.0, 1.0]])
        got = wall.evaluate_cartesian(ray_path)
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(expected, got)
        chex.assert_shape(got, ())


class TestRIS:
    def test_evaluate_cartesian(self):
        wall = RIS(points=jnp.array([[0.0, 0.0], [4.0, 0.0]]), phi=0.0)
        expected = jnp.array(0.0)
        ray_path = jnp.array([[0.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
        got = wall.evaluate_cartesian(ray_path)
        chex.assert_trees_all_close(expected, got)
        chex.assert_shape(got, ())
        ray_path = jnp.array([[0.0, 1.0], [2.0, 0.0], [4.0, 1.0]])
        got = wall.evaluate_cartesian(ray_path)
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(expected, got)
        chex.assert_shape(got, ())


class TestPath:
    def test_from_tx_objects_rx(self):
        wall = Wall(points=jnp.array([[0.0, 0.0], [2.0, 0.0]]))
        tx = Point(point=jnp.array([0.0, 1.0]))
        rx = Point(point=jnp.array([2.0, 1.0]))
        path = Path.from_tx_objects_rx(tx=tx.point, rx=rx.point, objects=[wall])
        chex.assert_trees_all_close(path.length(), 2.0 * jnp.sqrt(2.0))

    @path_cls
    def test_from_tx_objects_rx_no_object(self, path_cls: Type[Path]):
        tx = Point(point=jnp.array([0.0, 1.0]))
        rx = Point(point=jnp.array([2.0, 1.0]))
        path = path_cls.from_tx_objects_rx(tx=tx.point, rx=rx.point, objects=[])
        chex.assert_trees_all_close(path.length(), jnp.array(2.0))

    def test_path_length(self, key: Array):
        points = jax.random.uniform(key, (200, 2))
        expected = path_length(points)
        got = Path(points=points).length()
        chex.assert_trees_all_equal(expected, got)

    @approx
    def test_on_objects(self, approx: bool, key: Array):
        with enable_approx(approx), disable_jit():
            scene = Scene.random_uniform_scene(key, n_walls=5)
            path = Path.from_tx_objects_rx(
                scene.transmitters["tx_0"].point,
                scene.objects,  # type: ignore[arg-type]
                scene.receivers["rx_0"].point,
            )
            expected = true_value()
            got = path.on_objects(scene.objects)
            chex.assert_trees_all_close(expected, got, atol=1e-8)

            key = jax.random.split(key)[1]
            scene = Scene.random_uniform_scene(key, n_walls=5)
            expected = false_value()
            got = path.on_objects(scene.objects)
            chex.assert_trees_all_close(expected, got, atol=1e-8)

    @approx
    def test_intersects_with_objects(self, approx: bool, key: Array):
        with enable_approx(approx), disable_jit():
            scene = Scene.random_uniform_scene(key, n_walls=10)
            path = Path.from_tx_objects_rx(
                scene.transmitters["tx_0"].point,
                scene.objects,  # type: ignore[arg-type]
                scene.receivers["rx_0"].point,
            )
            path_candidate = jnp.arange(len(scene.objects), dtype=int)
            expected = true_value()
            # Very high probability that random paths intersect
            # at least one object in the scene.
            got = path.intersects_with_objects(scene.objects, path_candidate)
            chex.assert_trees_all_close(expected, got, atol=1e-8)

            scene = Scene.square_scene()
            path = Path.from_tx_objects_rx(
                scene.transmitters["tx"].point,
                scene.objects,  # type: ignore[arg-type]
                scene.receivers["rx"].point,
            )
            path_candidate = jnp.arange(len(scene.objects), dtype=int)
            expected = false_value()
            got = path.intersects_with_objects(scene.objects, path_candidate)
            chex.assert_trees_all_close(expected, got, atol=1e-8)

    @approx
    @path_cls
    def test_is_valid(self, approx: bool, path_cls: Type[Path]):
        with enable_approx(approx), disable_jit():
            scene = Scene.square_scene()
            path = path_cls.from_tx_objects_rx(
                scene.transmitters["tx"].point,
                scene.objects,  # type: ignore[arg-type]
                scene.receivers["rx"].point,
            )
            path_candidate = jnp.arange(len(scene.objects), dtype=int)
            interacting_objects = scene.get_interacting_objects(path_candidate)
            got = path.is_valid(scene.objects, path_candidate, interacting_objects)
            assert is_true(got)

    def test_plot(self, ax):
        wall = Wall(points=jnp.array([[0.0, 0.0], [2.0, 0.0]]))
        tx = Point(point=jnp.array([0.0, 1.0]))
        rx = Point(point=jnp.array([2.0, 1.0]))
        path = Path.from_tx_objects_rx(tx=tx.point, rx=rx.point, objects=[wall])
        _ = path.plot(ax)

    def test_bounding_box(self):
        expected = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 1.0],
            ]
        )
        wall = Wall(points=jnp.array([[0.0, 0.0], [2.0, 0.0]]))
        tx = Point(point=jnp.array([0.0, 1.0]))
        rx = Point(point=jnp.array([2.0, 1.0]))
        path = Path.from_tx_objects_rx(tx=tx.point, rx=rx.point, objects=[wall])
        got = path.bounding_box()
        chex.assert_trees_all_equal(expected, got)
        chex.assert_shape(got, (2, 2))


class TestImagePath:
    def test_path_loss_is_zero(self):
        scene = Scene.square_scene()
        got = ImagePath.from_tx_objects_rx(
            scene.transmitters["tx"].point,
            scene.objects,  # type: ignore[arg-type]
            scene.receivers["rx"].point,
        )
        chex.assert_trees_all_close(jnp.array(0.0), got.loss, atol=1e-13)


class TestFermatPath:
    def test_simple_reflection(self, seed: int, steps: int):
        wall = Wall(points=jnp.array([[0.0, 0.0], [2.0, 0.0]]))
        tx = Point(point=jnp.array([0.0, 1.0]))
        rx = Point(point=jnp.array([2.0, 1.0]))
        expected_points = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        got = FermatPath.from_tx_objects_rx(
            tx.point, [wall], rx.point, seed=seed, steps=steps
        )
        chex.assert_trees_all_close(expected_points, got.points, rtol=1e-2)
        chex.assert_shape(got.points, (3, 2))


class TestMinPath:
    def test_simple_reflection(self, seed: int, steps: int):
        wall = Wall(points=jnp.array([[0.0, 0.0], [2.0, 0.0]]))
        tx = Point(point=jnp.array([0.0, 1.0]))
        rx = Point(point=jnp.array([2.0, 1.0]))
        expected_loss = jnp.array(0.0)
        expected_points = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        got = MinPath.from_tx_objects_rx(
            tx.point, [wall], rx.point, seed=seed, steps=steps
        )
        chex.assert_trees_all_close(expected_points, got.points, rtol=1e-2)
        chex.assert_shape(got.points, (3, 2))
        chex.assert_trees_all_close(expected_loss, got.loss, atol=1e-4)
        chex.assert_shape(got.loss, ())
