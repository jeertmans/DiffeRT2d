from typing import TYPE_CHECKING, Any, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from differt2d.abc import Interactable
from differt2d.geometry import MinPath, Point
from differt2d.logic import less, logical_and, logical_not
from differt2d.scene import Scene

if TYPE_CHECKING:
    from jax import Array
else:
    Array = Any


def line_of_sight(
    tx: Point,
    objects: List[Interactable],
    rx_point: Array,
    tol: float = 1e-3,
) -> Array:
    rx = Point(point=rx_point)

    path_candidate = [0, len(objects) + 1]  # First and last
    interacting_objects = []  # type: ignore[var-annotated]

    path = MinPath.from_tx_objects_rx(tx, interacting_objects, rx)

    valid = path.on_objects(interacting_objects)
    valid = logical_and(
        valid, logical_not(path.intersects_with_objects(objects, path_candidate))
    )
    valid = logical_and(valid, less(path.loss, tol))

    return valid  # A true value means that LOS exists


def main():
    ax = plt.gca()
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    scene = Scene.basic_scene()
    scene.plot(ax)

    X, Y = scene.grid(n=300)

    grid = jnp.dstack((X, Y))

    vfunc = jax.vmap(
        jax.vmap(line_of_sight, in_axes=(None, None, 0)),
        in_axes=(None, None, 0),
    )

    Z = vfunc(scene.tx, scene.objects, grid)

    plt.pcolormesh(X, Y, Z)

    plt.show()


if __name__ == "__main__":
    main()
