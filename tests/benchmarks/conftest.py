import pytest
from jaxtyping import PRNGKeyArray

from differt2d.scene import Scene


@pytest.fixture(scope="session")
def scene(key: PRNGKeyArray) -> Scene:
    return Scene.basic_scene()
