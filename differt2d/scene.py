from typing import TYPE_CHECKING, Any, List, Union

import chex
import numpy as np
import rustworkx as rx

from .abc import Interactable, Plottable
from .geometry import MinPath, Point

if TYPE_CHECKING:
    from jax import Array
else:
    Array = Any


class Object(Interactable, Plottable):
    pass


@chex.dataclass
class Scene(Plottable):
    tx: Point
    rx: Point
    objects: List[Object]

    def plot(self, ax, *args, **kwargs) -> List[Any]:
        return [
            self.tx.plot(ax, *args, **kwargs),
            self.rx.plot(ax, *args, **kwargs),
        ] + [obj.plot(ax, *args, **kwargs) for obj in self.objects]

    def bounding_box(self) -> Array:
        bounding_boxes = [self.tx.bounding_box(), self.tx.bounding_box()] + [
            obj.bounding_box() for obj in self.objects
        ]
        bounding_boxes = ...
        pass

    def all_path_candidates(self, order=1):
        n = len(self.objects)
        matrix = np.ones((n + 2, n + 2))

        graph = rx.PyGraph.from_adjacency_matrix(matrix)

        return rx.all_simple_paths(graph, 0, n + 1, cutoff=order + 2)

    def all_paths(self, order=1, epsilon=1e-4):
        paths = []

        for path_candidate in self.all_path_candidates(order=order):
            objects = [self.objects[i - 1] for i in path_candidate[1:-1]]

            path = MinPath.from_tx_objects_rx(self.tx, objects, self.rx)

            if (
                path.on_objects(objects)
                and not path.intersects_with_objects(self.objects)
                and path.loss < epsilon
            ):
                paths.append(path)

        return paths
