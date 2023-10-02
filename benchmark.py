"""
The present files contains benchmarks to evaluate performance changes in the tool.

Requirements
------------

First, make sure ``pyperf`` is installed. If you have cloned
the repository, you can use ``poetry install --with test``.

Usage
-----

Running the following command will execute the benchmarks and save the results
for ``bench.json``: ``python benchmark.py -o bench.json``.

To compare multiple benchmark results, you can use
``python -m pyperf compare_to --table bench1.json bench2.json bench3.json ...``.
"""

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
