from functools import partial
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import rustworkx as rx

from .abc import Interactable, Plottable
from .geometry import MinPath, Point, Wall
from .logic import is_true, less, logical_and, logical_not

if TYPE_CHECKING:
    from dataclasses import dataclass

    from jax import Array
else:
    Array = Any
    from chex import dataclass


@partial(jax.jit, inline=True)
def power(path, path_candidate, objects):
    l1 = path.length()
    l2 = l1 * l1
    return 1 / (1.0 + l2)


@partial(jax.jit, inline=True)
def los_exists(path, path_candidate, objects):
    l1 = path.length()
    l2 = l1 * l1
    return 1 / (1.0 + l2)


# @partial(jax.jit, static_argnames=("objects", "function"))
def accumulate_at_location(
    tx: Point, objects, rx: Point, path_candidates, function
) -> Array:
    acc = jnp.array(0.0)
    tol = 1e-3

    for path_candidate in path_candidates:
        interacting_objects = [objects[i - 1] for i in path_candidate[1:-1]]

        path = MinPath.from_tx_objects_rx(tx, interacting_objects, rx)

        valid = path.on_objects(interacting_objects)
        valid = logical_and(valid, logical_not(path.intersects_with_objects(objects, path_candidate)))
        valid = logical_and(valid, less(path.loss, tol))

        acc += valid * function(path, path_candidate, interacting_objects)

    return acc


# @partial(jax.jit, static_argnames=("function"))
def _accumulate_at_location(
    tx: Point, objects, rx: Array, path_candidates, function
) -> Array:
    return accumulate_at_location(
        tx, objects, Point(point=rx), path_candidates, function
    )


@dataclass
class Scene(Plottable):
    """
    2D Scene made of objects, one emitting node, and one receiving node.
    """

    tx: Point
    """
    The emitting node.
    """
    rx: Point
    """
    The receiving node.
    """
    objects: Sequence[Union[Interactable, Plottable]]
    """
    The list of objects in the scene.
    """

    @classmethod
    def basic_scene(cls) -> "Scene":
        """
        Instantiates a basic scene with a main room,
        and a second inner room in the lower left corner,
        with a small entrance.

        :return: The scene.
        :rtype: Scene

        :EXAMPLES:

        >>> from differt2d.scene import Scene
        >>>
        >>> scene = Scene.basic_scene()
        >>> scene.bounding_box()
        Array([[0., 0.],
               [1., 1.]], dtype=float32)
        >>> len(scene.objects)
        7
        >>> scene.tx
        Point(point=Array([0.1, 0.1], dtype=float32))

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.basic_scene()
            scene.plot(ax)
            plt.show()
        """
        tx = Point(point=jnp.array([0.1, 0.1]))
        rx = Point(point=jnp.array([0.415, 0.2]))

        walls = [
            # Outer walls
            Wall(points=jnp.array([[0.0, 0.0], [1.0, 0.0]])),
            Wall(points=jnp.array([[1.0, 0.0], [1.0, 1.0]])),
            Wall(points=jnp.array([[1.0, 1.0], [0.0, 1.0]])),
            Wall(points=jnp.array([[0.0, 1.0], [0.0, 0.0]])),
            # Small room
            Wall(points=jnp.array([[0.4, 0.0], [0.4, 0.4]])),
            Wall(points=jnp.array([[0.4, 0.4], [0.3, 0.4]])),
            Wall(points=jnp.array([[0.1, 0.4], [0.0, 0.4]])),
        ]

        return Scene(tx=tx, rx=rx, objects=walls)

    def plot(self, ax, *args, **kwargs) -> List[Any]:
        return [
            self.tx.plot(ax, *args, color="red", **kwargs),
            self.rx.plot(ax, *args, color="green", **kwargs),
        ] + [
            obj.plot(ax, *args, **kwargs) for obj in self.objects  # type: ignore[union-attr]
        ]

    def bounding_box(self) -> Array:
        bounding_boxes_list = [self.tx.bounding_box(), self.rx.bounding_box()] + [
            obj.bounding_box() for obj in self.objects  # type: ignore[union-attr]
        ]
        bounding_boxes = jnp.dstack(bounding_boxes_list)

        return jnp.row_stack(
            [
                jnp.min(bounding_boxes[0, :, :], axis=1),
                jnp.max(bounding_boxes[1, :, :], axis=1),
            ]
        )

    def grid(self, n: int = 50) -> Tuple[Array, Array]:
        """
        Returns a (mesh) grid that overlays all objects in the scene.

        :param n: The number of sample along one axis.
        :type n: int
        :return: A tuple of (X, Y) coordinates.
        :rtype: ((n, n), (n, n)), typing.Tuple[jax.Array, jax.Array]
        """
        bounding_box = self.bounding_box()
        x = jnp.linspace(bounding_box[0, 0], bounding_box[1, 0], n)
        y = jnp.linspace(bounding_box[0, 1], bounding_box[1, 1], n)

        return jnp.meshgrid(x, y)

    def all_path_candidates(self, order: int = 1) -> List[List[int]]:
        """
        Returns all path candidates, from :attr:`tx` to :attr:`rx`,
        as a list of list of indices.

        Note that index 0 is for :attr:`tx`, and last index is for :attr:`rx`.

        :param order:
            The maximum order of the path, i.e., the number of interactions.
        :type order: int
        :return: The list of list of indices.
        :rtype: typing.List[typing.List[int]]
        """
        n = len(self.objects)
        matrix = np.ones((n + 2, n + 2))

        graph = rx.PyGraph.from_adjacency_matrix(matrix)

        return rx.all_simple_paths(graph, 0, n + 1, cutoff=order + 2)

    def all_paths(self, order: int = 1, tol: float = 1e-4) -> List[MinPath]:
        """
        Returns all valid paths from :attr:`tx` to :attr:`rx`,
        using the MPT method,
        see :class:`differt2d.geometry.MinPath`.

        :param order:
            The maximum order of the path, see :meth:`all_path_candidates`.
        :type order: int
        :param tol: The threshold tolerance for a path loss to be accepted.
        :type tol: float
        :return: The list of paths.
        :rtype: typing.List[MinPath]
        """
        paths = []

        for path_candidate in self.all_path_candidates(order=order):
            interacting_objects = [self.objects[i - 1] for i in path_candidate[1:-1]]

            path = MinPath.from_tx_objects_rx(self.tx, interacting_objects, self.rx)

            valid = path.on_objects(interacting_objects)
            valid = logical_and(
                valid,
                logical_not(path.intersects_with_objects(self.objects, path_candidate)),
            )
            valid = logical_and(valid, less(path.loss, tol))

            jax.debug.print("Path is valid: {v}, path={p}", v=valid, p=path)

            if is_true(valid):
                paths.append(path)

        return paths

    def accumulate_over_paths(
        self, function=power, order: int = 1, tol: float = 1e-4
    ) -> Array:
        """
        Returns all valid paths from :attr:`tx` to :attr:`rx`,
        using the MPT method,
        see :class:`differt2d.geometry.MinPath`.

        :param order:
            The maximum order of the path, see :meth:`all_path_candidates`.
        :type order: int
        :param tol: The threshold tolerance for a path loss to be accepted.
        :type tol: float
        :return: The list of paths.
        :rtype: typing.List[MinPath]
        """
        path_candidates = self.all_path_candidates(order=order)

        return accumulate_at_location(
            self.tx, self.objects, self.rx, path_candidates, function
        )

    def accumulate_on_grid(
        self, X, Y, function=power, order: int = 1, tol: float = 1e-4
    ) -> Array:
        """
        Returns all valid paths from :attr:`tx` to :attr:`rx`,
        using the MPT method,
        see :class:`differt2d.geometry.MinPath`.

        :param order:
            The maximum order of the path, see :meth:`all_path_candidates`.
        :type order: int
        :param tol: The threshold tolerance for a path loss to be accepted.
        :type tol: float
        :return: The list of paths.
        :rtype: typing.List[MinPath]
        """
        path_candidates = self.all_path_candidates(order=order)

        grid = jnp.dstack((X, Y))

        vacc = jax.vmap(
            jax.vmap(_accumulate_at_location, in_axes=(None, None, 0, None, None)),
            in_axes=(None, None, 0, None, None),
        )

        return vacc(self.tx, self.objects, grid, path_candidates, function)
