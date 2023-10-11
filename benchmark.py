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

import inspect
from functools import partial

import pyperf

from differt2d.geometry import FermatPath, ImagePath, MinPath
from differt2d.scene import Scene
from differt2d.utils import received_power

METHOD_TO_PATH_CLASS = {"image": ImagePath, "FPT": FermatPath, "MPT": MinPath}

scene = Scene.basic_scene()
X, Y = scene.grid(n=100)


def make_benchmark_name(*args):
    caller_name = inspect.stack()[1].function

    if caller_name.startswith("bench_"):
        caller_name = caller_name[6:]

    return "_".join([caller_name, *args])


def bench_accumulate_on_emitters_grid_over_paths(runner):
    def bench(approx):
        scene.accumulate_on_emitters_grid_over_paths(
            X, Y, fun=received_power, reduce_all=True, approx=approx
        ).block_until_ready()

    for approx in [False, True]:
        bench_name = make_benchmark_name("approx" if approx else "noapprox")
        func = partial(bench, approx=approx)
        runner.bench_func(bench_name, func)


def bench_path_method(runner):
    def bench(path_cls):
        path = path_cls.from_tx_objects_rx(
            scene.emitters["tx"].point, scene.objects, scene.receivers["rx"].point
        )
        path.loss.block_until_ready()

    for method, path_cls in METHOD_TO_PATH_CLASS.items():
        bench_name = make_benchmark_name(method)
        func = partial(bench, path_cls=path_cls)
        runner.bench_func(bench_name, func)


runner = pyperf.Runner()
bench_accumulate_on_emitters_grid_over_paths(runner)
bench_path_method(runner)
