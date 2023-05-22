from typing import Any, List, Union

import chex
import numpy as np
import rustworkx as rx

from .abc import Interactable, Plottable
from .geometry import Point, MinPath


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

    def all_path_candidates(self, order=1):
        n = len(self.objects)
        matrix = np.ones((n + 2, n + 2))

        graph = rx.PyGraph.from_adjacency_matrix(matrix)

        return rx.all_simple_paths(graph, 0, n + 1, cutoff=order + 2)

    def all_paths(self, order=1):

        paths = []

        for path_candidate in self.all_path_candidates(order=order):
            objects = [self.objects[i - 1] for i in path_candidate[1:-1]]

            path = MinPath.from_tx_objects_rx(self.tx, objects, self.rx)

            paths.append(path)

        return paths
