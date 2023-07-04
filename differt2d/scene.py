from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import rustworkx as rx

from .abc import Interactable, Plottable
from .geometry import FermatPath, MinPath, Path, Point, Wall
from .logic import is_true, less, logical_and, logical_not

if TYPE_CHECKING:
    from dataclasses import dataclass

    from jax import Array
    from matplotlib.artist import Artist
else:
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
        valid = logical_and(
            valid, logical_not(path.intersects_with_objects(objects, path_candidate))
        )
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
            _ = scene.plot(ax)
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

    def plot(
        self,
        ax,
        *args: Any,
        tx_args: Sequence = (),
        tx_kwargs: Dict[str, Any] = {},
        objects_args: Sequence = (),
        objects_kwargs: Dict[str, Any] = {},
        rx_args: Sequence = (),
        rx_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Union[Artist, List[Artist]]:
        """
        :param tx_args:
            Parameters to be passed to TX's plot function.
        :param tx_kwargs:
            Keyword parameters to be passed to TX's plot function.
        :param objects_args:
            Parameters to be passed to the objects' plot function.
        :param objects_kwargs:
            Keyword parameters to be passed to the objects' plot function.
        :param rx_args:
            Parameters to be passed to RX's plot function.
        :param rx_kwargs:
            Keyword parameters to be passed to RX's plot function.
        """
        tx_kwargs.setdefault("color", "blue")
        rx_kwargs.setdefault("color", "green")

        return [
            self.tx.plot(ax, *tx_args, *args, **tx_kwargs, **kwargs),
            self.rx.plot(ax, *rx_args, *args, **rx_kwargs, **kwargs),
        ] + [
            obj.plot(ax, *objects_args, *args, **objects_kwargs, **kwargs) for obj in self.objects  # type: ignore[union-attr]
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

        X, Y = jnp.meshgrid(x, y)
        return X, Y

    def all_path_candidates(
        self, min_order: int = 0, max_order: int = 1
    ) -> List[List[int]]:
        """
        Returns all path candidates, from :attr:`tx` to :attr:`rx`,
        as a list of list of indices.

        Note that index 0 is for :attr:`tx`, and last index is for :attr:`rx`.

        :param min_order:
            The minimum order of the path, i.e., the number of interactions.
        :param max_order:
            The maximum order of the path, i.e., the number of interactions.
        :type order: int
        :return: The list of list of indices.
        :rtype: typing.List[typing.List[int]]
        """
        n = len(self.objects)
        matrix = np.ones((n + 2, n + 2))

        graph = rx.PyGraph.from_adjacency_matrix(matrix)

        return rx.all_simple_paths(
            graph, 0, n + 1, min_depth=min_order + 2, cutoff=max_order + 2
        )

    def all_paths(
        self, tol: float = 1e-4, method: Literal["FPT", "MPT"] = "FPT", **kwargs: Any
    ) -> List[Path]:
        """
        Returns all valid paths from :attr:`tx` to :attr:`rx`,
        using the given method,
        see :class:`differt2d.geometry.FermatPath`
        and :class:`differt2d.geometry.MinPath`.

        :param tol: The threshold tolerance for a path loss to be accepted.
        :param method: Method to be used to find the path coordinates.
        :param kwargs:
            Keyword arguments to be passed to :meth:`all_path_candidates`.
        :return: The list of paths.
        """
        paths = []

        path_class: type[Path]
        if method == "FPT":
            path_class = FermatPath
        elif method == "MPT":
            path_class = MinPath
        else:
            raise ValueError(f"Unknown method name '{method}'")

        for path_candidate in self.all_path_candidates(**kwargs):
            interacting_objects = [self.objects[i - 1] for i in path_candidate[1:-1]]

            path = path_class.from_tx_objects_rx(self.tx, interacting_objects, self.rx)

            valid = path.on_objects(interacting_objects)
            valid = logical_and(
                valid,
                logical_not(path.intersects_with_objects(self.objects, path_candidate)),
            )

            if isinstance(path, MinPath):
                valid = logical_and(valid, less(path.loss, tol))

            jax.debug.print("Path is valid: {v}, path={p}", v=valid, p=path)

            if is_true(valid):
                paths.append(path)

        return paths

    def accumulate_over_paths(
        self, function=power, tol: float = 1e-4, **kwargs: Any
    ) -> Array:
        path_candidates = self.all_path_candidates(**kwargs)

        return accumulate_at_location(
            self.tx, self.objects, self.rx, path_candidates, function
        )

    def accumulate_on_grid(
        self, X, Y, function=power, tol: float = 1e-4, **kwargs
    ) -> Array:
        path_candidates = self.all_path_candidates(**kwargs)

        grid = jnp.dstack((X, Y))

        vacc = jax.vmap(
            jax.vmap(_accumulate_at_location, in_axes=(None, None, 0, None, None)),
            in_axes=(None, None, 0, None, None),
        )

        return vacc(self.tx, self.objects, grid, path_candidates, function)
