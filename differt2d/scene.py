from __future__ import annotations

import json
from enum import Enum
from itertools import product
from functools import partial, singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
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
import matplotlib.pyplot as plt
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

    emitters: Dict[str, Point]
    """
    The emitting node.
    """
    receivers: Dict[str, Point]
    """
    The receiving node.
    """
    objects: List[Union[Interactable, Plottable]]
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

        scene = cls(emitters=dict(tx=tx), receivers=dict(rx=rx), objects=walls)
        scene.emitters["tx"] = Point(point=scene.get_location(tx_loc))
        scene.receivers["rx"] = Point(point=scene.get_location(rx_loc))
        return scene

    @from_geojson.register(Readable)
    @classmethod
    def _(cls, fp: Readable, *args: Any, **kwargs: Any) -> "Scene":
        return cls.from_geojson(fp.read(), *args, **kwargs)

    def add_objects(self, objects: Sequence[Union[Interactable, Plottable]]) -> None:
        """
        Add objects to the scene.
        """
        self.objects.extend(objects)

    class SceneName(str, Enum):
        basic_scene = "basic_scene"
        square_scene = "square_scene"
        square_scene_with_obstacle = "square_scene_with_obstacle"

    @classmethod
    def from_scene_name(
        cls, scene_name: Union["Scene.SceneName", str], *args: Any, **kwargs: Any
    ) -> "Scene":
        """
        Generates a new scene from the given scene name.

        :param scene_name: The name of the scene.
        :param args:
            Positional arguments to be passed to the constructor.
        :param kwargs:
            Keyword arguments to be passed to the constructor.
        :return: The scene.
        """
        if isinstance(scene_name, cls.SceneName):
            scene_name_str = scene_name.value
        else:
            scene_name_str = scene_name

        return getattr(cls, scene_name_str)(*args, **kwargs)

    @classmethod
    def random_uniform_scene(
        cls,
        key: jax.random.KeyArray,
        *,
        n_emitters: int = 1,
        n_walls: int = 1,
        n_receivers: int = 1,
    ) -> "Scene":
        """
        Generates a random scene,
        drawing coordinates from a random distribution.

        :param key: The random key to be used.
        :param n_emitters: The number of emitters.
        :param n_walls: The number of walls.
        :param n_receivers: The number of receivers.
        :return: The scene.

        :Examples:

        .. code-block::

            import matplotlib.pyplot as plt
            import jax
            from differt2d.scene import Scene

            ax = plt.gca()
            key = jax.random.PRNGKey(1234)
            scene = Scene.random_uniform_scene(key, n_walls=5)
            #_ = scene.plot(ax)
            plt.show()
        """
        points = jax.random.uniform(key, (n_emitters + 2 * n_walls + n_receivers, 2))
        emitters = {f"tx_{i}": Point(point=points[i, :]) for i in range(n_emitters)}
        receivers = {
            f"rx_{i}": Point(point=points[-(i + 1), :]) for i in range(n_receivers)
        }

        walls = [
            Wall(points=points[2 * i + n_emitters : 2 * i + 2 + n_emitters, :])
            for i in range(n_walls)
        ]
        return cls(emitters=emitters, receivers=receivers, objects=walls)

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
        >>> scene.emitters["tx"]
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

        return cls(emitters=dict(tx=tx), receivers=dict(rx=rx), objects=walls)

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
        >>> scene.emitters["tx"]
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
        rx = Point(point=jnp.array([0.5, 0.6]))

        walls = [
            Wall(points=jnp.array([[0.0, 0.0], [1.0, 0.0]])),
            Wall(points=jnp.array([[1.0, 0.0], [1.0, 1.0]])),
            Wall(points=jnp.array([[1.0, 1.0], [0.0, 1.0]])),
            Wall(points=jnp.array([[0.0, 1.0], [0.0, 0.0]])),
        ]

        return Scene(emitters=dict(tx=tx), receivers=dict(rx=rx), objects=walls)

    @classmethod
    def square_scene_with_obstacle(cls, ratio: float = 0.1) -> "Scene":
        """
        Instantiates a square scene with one main room,
        and one square obstacle in the center.

        :param ratio: The ratio of the obstacle's side length to
            the room's side length.
        :return: The scene.

        :Examples:

        >>> from differt2d.scene import Scene
        >>>
        >>> scene = Scene.square_scene_with_obstacle()
        >>> scene.bounding_box()
        Array([[0., 0.],
               [1., 1.]], dtype=float32)
        >>> len(scene.objects)
        8
        >>> scene.emitters["tx"]
        Point(point=Array([0.2, 0.2], dtype=float32))

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            plt.show()
        """
        scene = Scene.square_scene()

        hl = 0.5 * ratio

        x0, x1 = 0.5 - hl, 0.5 + hl
        y0, y1 = 0.5 - hl, 0.5 + hl

        walls = [
            Wall(points=jnp.array([[x0, y0], [x1, y0]])),
            Wall(points=jnp.array([[x1, y0], [x1, y1]])),
            Wall(points=jnp.array([[x1, y1], [x0, y1]])),
            Wall(points=jnp.array([[x0, y1], [x0, y0]])),
        ]

        scene.add_objects(walls)

        return scene

    def plot(
        self,
        ax,
        *args: Any,
        emitters_args: Sequence = (),
        emitters_kwargs: Dict[str, Any] = {},
        objects_args: Sequence = (),
        objects_kwargs: Dict[str, Any] = {},
        receivers_args: Sequence = (),
        receivers_kwargs: Dict[str, Any] = {},
        annotate: bool = True,
        **kwargs: Any,
    ) -> Union[Artist, List[Artist]]:
        """
        :param emitters_args:
            Arguments to be passed to each emitter's plot function.
        :param tx_kwargs:
            Keyword arguments to be passed to each emitter's plot function.
        :param objects_args:
            Arguments to be passed to the each object' plot function.
        :param objects_kwargs:
            Keyword arguments to be passed to each object' plot function.
        :param receiver_args:
            Arguments to be passed to each receiver's plot function.
        :param receiver_kwargs:
            Keyword arguments to be passed to each receiver's plot function.
        :param annotate:
            If set, will annotate all emitters and receivers with their name,
            and append the corresponding artists
            to the end of the returned list.
        """
        emitters_kwargs.setdefault("color", "blue")
        receivers_kwargs.setdefault("color", "green")

        artists = (
            [
                emitter.plot(ax, *emitters_args, *args, **emitters_kwargs, **kwargs)
                for emitter in self.emitters.values()
            ]
            + [
                receiver.plot(ax, *receivers_args, *args, **receivers_kwargs, **kwargs)
                for receiver in self.receivers.values()
            ]
            + [
                obj.plot(ax, *objects_args, *args, **objects_kwargs, **kwargs) for obj in self.objects  # type: ignore[union-attr]
            ]
        )

        if annotate:
            if ax is None:
                ax = plt.gca()

            artists += [
                ax.annotate(e_key, emitter.point)
                for e_key, emitter in self.emitters.items()
            ] + [
                ax.annotate(r_key, receiver.point)
                for r_key, receiver in self.receivers.items()
            ]

        return artists

    def bounding_box(self) -> Array:
        bounding_boxes_list = (
            [emitter.bounding_box() for emitter in self.emitters.values()]
            + [receiver.bounding_box() for receiver in self.receivers.values()]
            + [obj.bounding_box() for obj in self.objects]  # type: ignore[union-attr]
        )
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

    def all_emitter_receiver_pairs(
        self,
    ) -> Generator[Tuple[Tuple[str, int], Tuple[str, int]]]:
        """
        Returns all possible pairs of (emitter, receiver) in the scene.

        Each pair ``P := ((KE, E), (KR, R))`` is made of the following:

        + ``KE`` the name of the emitter (key);
        + ``E`` the actual emitter :class:`Point<differt2d.geometry.Point>` (value);
        + ``KR`` the name of the receiver (key);
        + ``R`` the actual receiver :class:`Point<differt2d.geometry.Point>` (value).

        :return: A generator of all possible pairs.
        """
        return product(self.emitters.items(), self.receivers.items())

    def all_path_candidates(
        self, min_order: int = 0, max_order: int = 1
    ) -> List[List[int]]:
        """
        Returns all path candidates, from any of the :attr:`emitters`
        to any of the :attr:`receivers`,
        as a list of list of indices.

        Note that index 0 is for :attr:`emitters`,
        and last index is for :attr:`receivers`.

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
    ) -> Dict[Tuple[str, str], Path]:
        """
        Returns all valid paths from any of the :attr:`emitters`
        to any of the :attr:`receivers`,
        using the given method,
        see, :class:`differt2d.geometry.ImagePath`
        :class:`differt2d.geometry.FermatPath`
        and :class:`differt2d.geometry.MinPath`.

        :param tol: The threshold tolerance for a path loss to be accepted.
        :param method: Method to be used to find the path coordinates.
        :param kwargs:
            Keyword arguments to be passed to :meth:`all_path_candidates`.
        :return: The list of paths, as a mapping with
            (emitter, receiver) names as entries.
        """
        paths = {}

        path_candidates = self.all_path_candidates(**kwargs)

        for (e_key, emitter), (r_key, receiver) in self.all_emitter_receiver_pairs():
            for path_candidate in path_candidates:
                interacting_objects: List[Interactable] = [
                    self.objects[i - 1] for i in path_candidate[1:-1]  # type: ignore
                ]

                path = method.from_tx_objects_rx(emitter, interacting_objects, receiver)

                valid = path.on_objects(interacting_objects)

                valid = logical_and(
                    valid,
                    logical_not(
                        path.intersects_with_objects(self.objects, path_candidate)
                    ),
                )

                valid = logical_and(valid, less(path.loss, tol))

                if is_true(valid):
                    paths.setdefault((e_key, r_key), []).append(path)

        return paths

    def accumulate_over_paths(self, function=power, **kwargs: Any) -> Array:
        """
        Accumulates some function evaluated for each path in the scene.

        :param function: The function to accumulate.
        """
        path_candidates = self.all_path_candidates(**kwargs)

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
