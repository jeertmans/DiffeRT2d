import pyperf

from differt2d.scene import Scene
from differt2d.utils import received_power

scene = Scene.basic_scene()
X, Y = scene.grid(n=300)


def bench_accumulate_on_emitters_grid_over_paths():
    scene.accumulate_on_emitters_grid_over_paths(X, Y, fun=received_power)


def bench_accumulate_on_receivers_grid_over_paths():
    scene.accumulate_on_receivers_grid_over_paths(X, Y, fun=received_power)


runner = pyperf.Runner()
runner.bench_func(
    "accumulate_on_emitters_grid_over_paths",
    bench_accumulate_on_emitters_grid_over_paths,
)
runner.bench_func(
    "accumulate_on_receivers_grid_over_paths",
    bench_accumulate_on_receivers_grid_over_paths,
)
