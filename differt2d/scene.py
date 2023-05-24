from typing import TYPE_CHECKING, Any, List, Tuple

import chex
import jax.numpy as jnp
import numpy as np
import rustworkx as rx

from .abc import Interactable, Plottable
from .geometry import MinPath, Point
from .logic import is_true, less, logical_and, logical_not

if TYPE_CHECKING:
    from jax import Array
else:
    Array = Any


class Object(Interactable, Plottable):
    """
    Abstract class for plottable and interactable objects.
    """

    pass


@chex.dataclass
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
    objects: List[Object]
    """
    The list of objects in the scene.
    """

    def plot(self, ax, *args, **kwargs) -> List[Any]:
        return [
            self.tx.plot(ax, *args, **kwargs),
            self.rx.plot(ax, *args, **kwargs),
        ] + [obj.plot(ax, *args, **kwargs) for obj in self.objects]

    def bounding_box(self) -> Array:
        bounding_boxes = [self.tx.bounding_box(), self.rx.bounding_box()] + [
            obj.bounding_box() for obj in self.objects
        ]
        bounding_boxes = jnp.dstack(bounding_boxes)

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
        :rtype: ((n, n), (n, n)), Tuple[Array, Array]
        """
        bounding_box = self.bounding_box()
        x = jnp.linspace(bounding_box[0, 0], bounding_box[1, 0], n)
        y = jnp.linspace(bounding_box[1, 0], bounding_box[1, 1], n)

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
        :rtype: List[list[int]]
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
        :rtype: List[MinPath]
        """
        paths = []

        for path_candidate in self.all_path_candidates(order=order):
            objects = [self.objects[i - 1] for i in path_candidate[1:-1]]

            path = MinPath.from_tx_objects_rx(self.tx, objects, self.rx)

            valid = path.on_objects(objects)
            valid = logical_and(
                valid, logical_not(path.intersects_with_objects(self.objects))
            )
            valid = logical_and(valid, less(path.loss, tol))

            if is_true(valid):
                paths.append(path)

        return paths
