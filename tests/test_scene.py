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
        scene = Scene.random_uniform_scene(key, n)

        assert len(scene.objects) == n
