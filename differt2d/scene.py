from __future__ import annotations

import json
from functools import partial, singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import numpy as np
import rustworkx as rx

from .abc import LOC, Interactable, Plottable
from .geometry import FermatPath, ImagePath, MinPath, Path, Point, Wall
from .logic import is_true, less, logical_and, logical_not

if TYPE_CHECKING:  # pragma: no cover
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


# @partial(jax.jit, static_argnames=("objects", "function"))
def accumulate_at_location(
    tx: Point, objects, rx: Point, path_candidates, function
) -> Array:
    acc = jnp.array(0.0)
    tol = 1e-3

    for path_candidate in path_candidates:
        interacting_objects = [objects[i - 1] for i in path_candidate[1:-1]]

        path = ImagePath.from_tx_objects_rx(tx, interacting_objects, rx)

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


def reduce_along_path_candidates(f, path_candidate):
    pass


@jax.jit
def select_objects_from_path_candidate(objects: List[Interactable], path_candidate: List[int]) -> List[Interactable]:
    """
    Returns the selection of objects from the given path candidate.

    :param object: The list of all objects.
    :param path_candidate: A path candidate,
        as returned by ...
    :return: The selection of objects.
    """
    return [objects[i - 1] for i in path_candidate[1:-1]]


S = Union[str, bytes, bytearray]


@runtime_checkable
class Readable(Protocol):
    def read(self) -> S:
        pass


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

    @singledispatchmethod
    @classmethod
    def from_geojson(
        cls, s_or_fp: Union[S, Readable], tx_loc: LOC = "NW", rx_loc: LOC = "SE"
    ) -> "Scene":
        r"""
        Creates a scene from a GEOJSON file, generating one Wall per
        line segment. TX and RX positions are located on the corner
        of the bounding box.

        :param s_or_fp: Source from which to read the GEOJSON object,
            either a string-like or a file-like object.
        :param tx_loc: Where to place TX, see
            :meth:`Plottable.get_location<differt2d.abc.Plottable.get_location>`.
        :param rx_loc: Where to place RX, see
            :meth:`Plottable.get_location<differt2d.abc.Plottable.get_location>`.
        :return: The scene representation of the GEOJSON.

        :Examples:

        The following example was obtained from https://overpass-turbo.eu/
        using following query:

        .. code::

            [out:json][timeout:30];(
            way["building"](50.66815414931746,4.624882042407989,50.66856810072477,4.6256572008132935);
            relation["building"]["type"="multipolygon"](50.66815414931746,4.624882042407989,50.66856810072477,4.6256572008132935);
            );out;>;out qt;


        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene
            s = r'''
            {
              "type": "FeatureCollection",
              "generator": "overpass-ide",
              "copyright": "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL.",
              "timestamp": "2023-07-27T15:01:41Z",
              "features": [
                {
                  "type": "Feature",
                  "properties": {
                    "@id": "way/492286203",
                    "building": "yes",
                    "building:levels": "1"
                  },
                  "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                      [
                        [
                          4.6251085,
                          50.6682201
                        ],
                        [
                          4.6251756,
                          50.6682511
                        ],
                        [
                          4.625094,
                          50.6683219
                        ],
                        [
                          4.6250833,
                          50.6683312
                        ],
                        [
                          4.6250217,
                          50.6683016
                        ],
                        [
                          4.6250217,
                          50.6682916
                        ],
                        [
                          4.6251085,
                          50.6682201
                        ]
                      ]
                    ]
                  },
                  "id": "way/492286203"
                },
                {
                  "type": "Feature",
                  "properties": {
                    "@id": "way/492286204",
                    "addr:housenumber": "1",
                    "addr:street": "Avenue Georges Lemaître",
                    "building": "yes",
                    "building:levels": "2"
                  },
                  "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                      [
                        [
                          4.6250343,
                          50.6683956
                        ],
                        [
                          4.6251091,
                          50.668334
                        ],
                        [
                          4.6252786,
                          50.6684176
                        ],
                        [
                          4.6253134,
                          50.6683623
                        ],
                        [
                          4.6252795,
                          50.6683505
                        ],
                        [
                          4.6253126,
                          50.6682971
                        ],
                        [
                          4.6253341,
                          50.6683029
                        ],
                        [
                          4.6253868,
                          50.668224
                        ],
                        [
                          4.6254999,
                          50.6682531
                        ],
                        [
                          4.6254757,
                          50.6682932
                        ],
                        [
                          4.6255124,
                          50.6683029
                        ],
                        [
                          4.625476,
                          50.6683594
                        ],
                        [
                          4.6254341,
                          50.6683482
                        ],
                        [
                          4.6254087,
                          50.6683895
                        ],
                        [
                          4.6253801,
                          50.668382
                        ],
                        [
                          4.6253402,
                          50.6684467
                        ],
                        [
                          4.6253647,
                          50.6684615
                        ],
                        [
                          4.6253092,
                          50.6685068
                        ],
                        [
                          4.6251988,
                          50.6684496
                        ],
                        [
                          4.6251799,
                          50.6684664
                        ],
                        [
                          4.6250343,
                          50.6683956
                        ]
                      ]
                    ]
                  },
                  "id": "way/492286204"
                }
              ]
            }'''

            ax = plt.gca()
            scene = Scene.from_geojson(s)
            _ = scene.plot(ax)
        """
        raise NotImplementedError(f"Unsupported type {type(s_or_fp)}")

    @from_geojson.register(str)
    @from_geojson.register(bytes)
    @from_geojson.register(bytearray)
    @classmethod
    def _(cls, s: S, tx_loc: LOC = "NW", rx_loc: LOC = "SE") -> "Scene":
        dictionary = json.loads(s)

        features = dictionary.get("features", [])

        walls = []

        for feature in features:
            if geometry := feature.get("geometry", None):
                _type = geometry["type"]
                coordinates = geometry["coordinates"][0]
                n = len(coordinates)

                if _type == "Polygon":
                    for i in range(n):
                        points = jnp.row_stack(
                            [coordinates[i - 1], coordinates[i]], dtype=float
                        )
                        wall = Wall(points=points)
                        walls.append(wall)

        if len(walls) > 0:
            tx = Point(point=walls[0].origin())
            rx = Point(point=walls[0].dest())

        else:
            tx = Point(point=jnp.array([0.0, 0.0]))
            rx = Point(point=jnp.array([1.0, 1.0]))

        scene = cls(tx=tx, rx=rx, objects=walls)
        scene.tx = Point(point=scene.get_location(tx_loc))
        scene.rx = Point(point=scene.get_location(rx_loc))
        return scene

    @from_geojson.register(Readable)
    @classmethod
    def _(cls, fp: Readable, *args: Any, **kwargs: Any) -> "Scene":
        return cls.from_geojson(fp.read(), *args, **kwargs)

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "n"))
    def random_uniform_scene(cls, key: jax.random.KeyArray, n: int) -> "Scene":
        """
        Generates a random scene with ``n`` walls,
        drawing coordinates from a random distribution.

        :Examples:

        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            import jax
            from differt2d.scene import Scene

            ax = plt.gca()
            key = jax.random.PRNGKey(134)
            scene = Scene.random_uniform_scene(key, 5)
            _ = scene.plot(ax)
            plt.show()
        """
        points = jax.random.uniform(key, (2 * n + 2, 2))
        tx = Point(point=points[+0, :])
        rx = Point(point=points[-1, :])

        walls = [Wall(points=points[2 * i + 1 : 2 * i + 3, :]) for i in range(n)]
        return cls(tx=tx, rx=rx, objects=walls)

    @classmethod
    def basic_scene(cls) -> "Scene":
        """
        Instantiates a basic scene with a main room,
        and a second inner room in the lower left corner,
        with a small entrance.

        :return: The scene.

        :Examples:

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
        rx = Point(point=jnp.array([0.302, 0.2147]))

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

        return cls(tx=tx, rx=rx, objects=walls)

    @classmethod
    def square_scene(cls) -> "Scene":
        """
        Instantiates a square scene with one main room.

        :return: The scene.

        :Examples:

        >>> from differt2d.scene import Scene
        >>>
        >>> scene = Scene.square_scene()
        >>> scene.bounding_box()
        Array([[0., 0.],
               [1., 1.]], dtype=float32)
        >>> len(scene.objects)
        4
        >>> scene.tx
        Point(point=Array([0.2, 0.2], dtype=float32))

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            plt.show()
        """
        tx = Point(point=jnp.array([0.2, 0.2]))
        rx = Point(point=jnp.array([0.5, 0.5]))

        walls = [
            Wall(points=jnp.array([[0.0, 0.0], [1.0, 0.0]])),
            Wall(points=jnp.array([[1.0, 0.0], [1.0, 1.0]])),
            Wall(points=jnp.array([[1.0, 1.0], [0.0, 1.0]])),
            Wall(points=jnp.array([[0.0, 1.0], [0.0, 0.0]])),
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
            Arguments to be passed to TX's plot function.
        :param tx_kwargs:
            Keyword arguments to be passed to TX's plot function.
        :param objects_args:
            Arguments to be passed to the objects' plot function.
        :param objects_kwargs:
            Keyword arguments to be passed to the objects' plot function.
        :param rx_args:
            Arguments to be passed to RX's plot function.
        :param rx_kwargs:
            Keyword arguments to be passed to RX's plot function.
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
        self,
        tol: float = 1e-2,
        method: Type[Path] = ImagePath,
        **kwargs: Any,
    ) -> List[Path]:
        """
        Returns all valid paths from :attr:`tx` to :attr:`rx`,
        using the given method,
        see, :class:`differt2d.geometry.ImagePath`
        :class:`differt2d.geometry.FermatPath`
        and :class:`differt2d.geometry.MinPath`.

        :param tol: The threshold tolerance for a path loss to be accepted.
        :param method: Method to be used to find the path coordinates.
        :param kwargs:
            Keyword arguments to be passed to :meth:`all_path_candidates`.
        :return: The list of paths.
        """
        paths = []

        for path_candidate in self.all_path_candidates(**kwargs):
            interacting_objects = select_objects_from_path_candidate(self.objects, path_candidate)
            valid = check_path(path, 

            path = method.from_tx_objects_rx(self.tx, interacting_objects, self.rx)

            valid = path.on_objects(interacting_objects)

            valid = logical_and(
                valid,
                logical_not(path.intersects_with_objects(self.objects, path_candidate)),
            )

            valid = logical_and(valid, less(path.loss, tol))

            if is_true(valid):
                paths.append(path)

        return paths

    def reduce_along_paths(self, function, *args, **kwargs):
        

    def accumulate_over_paths(self, function=power, **kwargs: Any) -> Array:
        """
        Accumulates some function evaluated for each path in the scene.

        :param function: The function to accumulate.
        """
        path_candidates = self.all_path_candidates(**kwargs)

        return reduce_over_paths(f, `

        return accumulate_at_location(
            self.tx, self.objects, self.rx, path_candidates, function
        )

    def accumulate_on_grid(
        self, X, Y, function=power, min_order: int = 0, max_order: int = 1, **kwargs
    ) -> Array:
        path_candidates = self.all_path_candidates(
            min_order=min_order, max_order=max_order
        )

        grid = jnp.dstack((X, Y))

        vacc = jax.vmap(
            jax.vmap(_accumulate_at_location, in_axes=(None, None, 0, None, None)),
            in_axes=(None, None, 0, None, None),
        )

        return vacc(self.tx, self.objects, grid, path_candidates, function, **kwargs)
