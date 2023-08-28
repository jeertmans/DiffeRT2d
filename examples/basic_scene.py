from functools import partial
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from differt2d.abc import LocEnum
from differt2d.logic import enable_approx
from differt2d.scene import Scene

EPS = jnp.finfo(float).eps


@partial(jax.jit, inline=True)
def power(path, path_candidate, objects):
    l1 = path.length()
    l2 = l1 * l1
    c = 0.5  # Power attenuation from one wall
    n = len(path_candidate) - 2  # Number of walls
    a = c**n
    p = a / (EPS + l2)

    return (p - 1.0) ** 2


def main(
    scene_name: Scene.SceneName = Scene.SceneName.basic_scene,
    file: Optional[Path] = None,
    resolution: int = 150,
    min_order: int = 0,
    max_order: int = 1,
    approx: bool = True,
    tx_loc: LocEnum = LocEnum.C,
    rx_loc: LocEnum = LocEnum.S,
    show_paths: bool = True,
    log_scale: bool = True,
):
    ax = plt.gca()

    if file:
        scene = Scene.from_geojson(
            file.read_text(), tx_loc=tx_loc.value, rx_loc=rx_loc.value
        )
    else:
        scene = Scene.from_scene_name(scene_name)
        scene.rx.point = jnp.array([0.5, 0.5])

    scene.plot(ax)

    with enable_approx(approx):
        if show_paths:
            for path in scene.all_paths(min_order=min_order, max_order=max_order):
                path.plot(ax)

        X, Y = scene.grid(n=resolution)

        Z = scene.accumulate_on_grid(
            X, Y, function=power, min_order=min_order, max_order=max_order
        )
        print(Z)

        if log_scale:
            Z = jnp.log1p(Z)

        plt.pcolormesh(X, Y, Z)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
