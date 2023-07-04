import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
from jax import disable_jit

from differt2d.geometry import FermatPath, MinPath, Point, Ray, Wall
from differt2d.logic import enable_approx, is_false, is_true

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


@pytest.fixture
def ax():
    return plt.gca()


@pytest.fixture
def seed():
    return 1234


@pytest.fixture
def steps():
    return 100


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
        chex.assert_tree_all_close(expected, got)
        chex.assert_shape(got, ())
        ray_path = jnp.array([[0.0, 1.0], [2.1, 0.0], [4.0, 1.0]])
        got = wall.evaluate_cartesian(ray_path)
        with pytest.raises(AssertionError):
            chex.assert_tree_all_close(expected, got)
        chex.assert_shape(got, ())


class TestFermatPath:
    @approx
    def test_simple_reflection(self, approx: bool, seed: int, steps: int):
        wall = Wall(points=jnp.array([[0.0, 0.0], [2.0, 0.0]]))
        tx = Point(point=jnp.array([0.0, 1.0]))
        rx = Point(point=jnp.array([2.0, 1.0]))
        expected_points = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])

        with enable_approx(approx), disable_jit():
            got = FermatPath.from_tx_objects_rx(tx, [wall], rx, seed=seed, steps=steps)
            chex.assert_tree_all_close(expected_points, got.points, rtol=1e-2)
            chex.assert_shape(got.points, (3, 2))


class TestMinPath:
    @approx
    def test_simple_reflection(self, approx: bool, seed: int, steps: int):
        wall = Wall(points=jnp.array([[0.0, 0.0], [2.0, 0.0]]))
        tx = Point(point=jnp.array([0.0, 1.0]))
        rx = Point(point=jnp.array([2.0, 1.0]))
        expected_loss = jnp.array(0.0)
        expected_points = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])

        with enable_approx(approx), disable_jit():
            got = MinPath.from_tx_objects_rx(tx, [wall], rx, seed=seed, steps=steps)
            chex.assert_tree_all_close(expected_points, got.points, rtol=1e-2)
            chex.assert_shape(got.points, (3, 2))
            chex.assert_tree_all_close(expected_loss, got.loss, atol=1e-4)
            chex.assert_shape(got.loss, ())
