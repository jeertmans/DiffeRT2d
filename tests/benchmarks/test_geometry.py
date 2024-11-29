import pytest
from jaxtyping import PRNGKeyArray
from pytest_benchmark.fixture import BenchmarkFixture

from differt2d.geometry import FermatPath, ImagePath, MinPath, Path
from differt2d.scene import Scene


@pytest.mark.parametrize("n", [5, 25, 50])
@pytest.mark.parametrize("path_cls", [Path, ImagePath, FermatPath, MinPath])
def test_accumulate_on_transmitters_grid_over_paths(
    n: int,
    path_cls: type[Path],
    scene: Scene,
    key: PRNGKeyArray,
    benchmark: BenchmarkFixture,
) -> None:
    _X, _Y = scene.grid(n)

    _ = benchmark(
        path_cls.from_tx_objects_rx(
            scene.transmitters["tx"],
            scene.objects,
            scene.receivers["rx"],
            key=key,
        ).loss.block_until_ready,  # type: ignore[reportAttributeAccessIssue]
    )
