import chex
import jax.numpy as jnp
import pytest

from differt2d.scene import Scene


class TestScene:
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
