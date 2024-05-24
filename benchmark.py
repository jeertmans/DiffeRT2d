"""
The present files contains benchmarks to evaluate performance changes in the tool.

Requirements
------------

First, make sure ``pyperf`` is installed. If you have cloned
the repository, you can use ``rye sync``.

Usage
-----

Running the following command will execute the benchmarks and save the results
for ``bench.json``: ``python benchmark.py -o bench.json``.

To compare multiple benchmark results, you can use
``python -m pyperf compare_to --table bench1.json bench2.json bench3.json ...``.
"""

import inspect
from functools import partial

import jax
import pyperf
from jaxtyping import Array, Float

from differt2d.geometry import FermatPath, ImagePath, MinPath, Path
from differt2d.scene import Scene
from differt2d.utils import received_power

METHOD_TO_PATH_CLASS = {"image": ImagePath, "FPT": FermatPath, "MPT": MinPath}

key = jax.random.PRNGKey(1234)
scene = Scene.basic_scene()
n = 100
X, Y = scene.grid(n=n)


def make_benchmark_name(*args: str) -> str:
    caller_name = inspect.stack()[1].function

    if caller_name.startswith("bench_"):
        caller_name = caller_name[6:]

    return "_".join([caller_name, *args])


def bench_accumulate_on_transmitters_grid_over_paths(runner: pyperf.Runner) -> None:
    def bench(approx: bool) -> None:
        array: Float[Array, "n n"] = scene.accumulate_on_transmitters_grid_over_paths(  # type: ignore[reportGeneralTypeIssues]
            X, Y, fun=received_power, reduce_all=True, approx=approx, key=key
        )
        array.block_until_ready()

    for approx in [False, True]:
        bench_name = make_benchmark_name("approx" if approx else "noapprox")
        func = partial(bench, approx=approx)
        runner.bench_func(bench_name, func)


def bench_path_method(runner: pyperf.Runner) -> None:
    def bench(path_cls: type[Path]) -> None:
        path = path_cls.from_tx_objects_rx(
            scene.transmitters["tx"],
            scene.objects,
            scene.receivers["rx"],
            key=key,
        )
        path.loss.block_until_ready()

    for method, path_cls in METHOD_TO_PATH_CLASS.items():
        bench_name = make_benchmark_name(method)
        func = partial(bench, path_cls=path_cls)
        runner.bench_func(bench_name, func)


runner = pyperf.Runner()
bench_accumulate_on_transmitters_grid_over_paths(runner)
bench_path_method(runner)
