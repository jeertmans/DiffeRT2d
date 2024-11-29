import pytest
from jaxtyping import PRNGKeyArray
from pytest_benchmark.fixture import BenchmarkFixture

from differt2d.scene import Scene
from differt2d.utils import received_power


@pytest.mark.parametrize("n", [5, 25, 50])
@pytest.mark.parametrize("approx", [False, True])
def test_accumulate_on_transmitters_grid_over_paths(
    n: int,
    approx: bool,
    scene: Scene,
    key: PRNGKeyArray,
    benchmark: BenchmarkFixture,
) -> None:
    X, Y = scene.grid(n)

    _ = benchmark(
        lambda: scene.accumulate_on_transmitters_grid_over_paths(
            X,
            Y,
            fun=received_power,
            reduce_all=True,
            approx=approx,
            key=key,
        ).block_until_ready(),  # type: ignore[reportAttributeAccessIssue]
    )
