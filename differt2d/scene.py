"""Scenes for tracing rays between transmitters and receivers."""

from __future__ import annotations

__all__ = ["Scene", "SceneName"]

import json
from functools import partial, singledispatchmethod
from itertools import groupby, product
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import differt_core
import jax
import jax.numpy as jnp

from .abc import Interactable, Loc, Object, Plottable
from .geometry import ImagePath, Path, Point, Wall, closest_point
from .logic import is_true

if TYPE_CHECKING:  # pragma: no cover
    from dataclasses import dataclass

    from jax import Array
    from matplotlib.artist import Artist

    PathFun = Callable[[Point, Point, Path, List[Interactable]], Array]
else:
    from chex import dataclass


S = Union[str, bytes, bytearray]
SceneName = Literal[
    "basic_scene",
    "square_scene",
    "square_scene_with_obstacle",
    "square_scene_with_wall",
]
"""Literal type for all valid scene names."""


@runtime_checkable
class Readable(Protocol):
    def read(self) -> S:
        pass  # pragma: no cover


@dataclass
class Scene(Plottable):
    """2D Scene made of objects, one or more transmitting node(s), and one or more
    receiving node(s)."""

    transmitters: Dict[str, Point]
    """The transmitting nodes."""
    receivers: Dict[str, Point]
    """The receiving nodes."""
    objects: List[Object]
    """The list of objects in the scene."""

    @singledispatchmethod
    @classmethod
    def from_geojson(
        cls, s_or_fp: Union[S, Readable], tx_loc: Loc = "NW", rx_loc: Loc = "SE"
    ) -> "Scene":
        r"""
        Creates a scene from a GEOJSON file, generating one Wall per line segment. TX
        and RX positions are located on the corner of the bounding box.

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
    def _(cls, s: S, tx_loc: Loc = "NW", rx_loc: Loc = "SE") -> "Scene":
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
                        points = jnp.array(
                            [coordinates[i - 1], coordinates[i]], dtype=float
                        )
                        wall = Wall(points=points)
                        walls.append(wall)

        if len(walls) > 0:
            tx = Point(point=walls[0].origin())
            rx = Point(point=walls[0].dest())

        else:  # pragma: no cover
            tx = Point(point=jnp.array([0.0, 0.0]))
            rx = Point(point=jnp.array([1.0, 1.0]))

        scene = cls(transmitters=dict(tx=tx), receivers=dict(rx=rx), objects=walls)  # type: ignore[arg-type]
        scene.transmitters["tx"] = Point(point=scene.get_location(tx_loc))
        scene.receivers["rx"] = Point(point=scene.get_location(rx_loc))
        return scene

    @from_geojson.register(Readable)
    @classmethod
    def _(cls, fp: Readable, *args: Any, **kwargs: Any) -> "Scene":
        return cls.from_geojson(fp.read(), *args, **kwargs)

    def add_objects(self, objects: Sequence[Object]) -> None:
        """Add objects to the scene."""
        self.objects.extend(objects)

    @classmethod
    def from_scene_name(
        cls, scene_name: SceneName, *args: Any, **kwargs: Any
    ) -> "Scene":
        """
        Generates a new scene from the given scene name.

        :param scene_name: The name of the scene.
        :param args: Positional arguments to be passed to the constructor.
        :param kwargs: Keyword arguments to be passed to the constructor.
        :return: The scene.
        """
        return getattr(cls, scene_name)(*args, **kwargs)

    @classmethod
    def random_uniform_scene(
        cls,
        key: Array,
        *,
        n_transmitters: int = 1,
        n_walls: int = 1,
        n_receivers: int = 1,
    ) -> "Scene":
        """
        Generates a random scene, drawing coordinates from a random distribution.

        :param key: The random key to be used.
        :param n_transmitters: The number of transmitters.
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
            _ = scene.plot(ax)
            plt.show()
        """
        points = jax.random.uniform(
            key, (n_transmitters + 2 * n_walls + n_receivers, 2)
        )
        transmitters = {
            f"tx_{i}": Point(point=points[i, :]) for i in range(n_transmitters)
        }
        receivers = {
            f"rx_{i}": Point(point=points[-(i + 1), :]) for i in range(n_receivers)
        }

        walls: List[Object] = [
            Wall(points=points[2 * i + n_transmitters : 2 * i + 2 + n_transmitters, :])
            for i in range(n_walls)
        ]
        return cls(transmitters=transmitters, receivers=receivers, objects=walls)

    @classmethod
    def basic_scene(
        cls,
        *,
        tx_coords: Array = jnp.array([0.1, 0.1]),
        rx_coords: Array = jnp.array([0.302, 0.2147]),
    ) -> "Scene":
        """
        Instantiates a basic scene with a main room, and a second inner room in the
        lower left corner, with a small entrance.

        :param tx_coords: The transmitter's coordinates, array-like, (2,).
        :param rx_coords: The receiver's coordinates, array-like, (2,).
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
        >>> scene.transmitters["tx"]
        Point(point=Array([0.1, 0.1], dtype=float32))

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.basic_scene()
            _ = scene.plot(ax)
            plt.show()
        """
        tx = Point(point=jnp.asarray(tx_coords, dtype=float))
        rx = Point(point=jnp.asarray(rx_coords, dtype=float))

        walls: List[Object] = [
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

        return cls(transmitters=dict(tx=tx), receivers=dict(rx=rx), objects=walls)

    @classmethod
    def square_scene(
        cls,
        *,
        tx_coords: Array = jnp.array([0.2, 0.2]),
        rx_coords: Array = jnp.array([0.5, 0.6]),
    ) -> "Scene":
        """
        Instantiates a square scene with one main room.

        :param tx_coords: The transmitter's coordinates, array-like, (2,).
        :param rx_coords: The receiver's coordinates, array-like, (2,).
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
        >>> scene.transmitters["tx"]
        Point(point=Array([0.2, 0.2], dtype=float32))

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            plt.show()
        """
        tx = Point(point=jnp.asarray(tx_coords, dtype=float))
        rx = Point(point=jnp.asarray(rx_coords, dtype=float))

        walls: List[Object] = [
            Wall(points=jnp.array([[0.0, 0.0], [1.0, 0.0]])),
            Wall(points=jnp.array([[1.0, 0.0], [1.0, 1.0]])),
            Wall(points=jnp.array([[1.0, 1.0], [0.0, 1.0]])),
            Wall(points=jnp.array([[0.0, 1.0], [0.0, 0.0]])),
        ]

        return Scene(transmitters=dict(tx=tx), receivers=dict(rx=rx), objects=walls)

    @classmethod
    def square_scene_with_wall(
        cls,
        ratio: float = 0.1,
        *,
        tx_coords: Array = jnp.array([0.2, 0.5]),
        rx_coords: Array = jnp.array([0.8, 0.5]),
    ) -> "Scene":
        """
        Instantiates a square scene with one main room, and vertical wall in the middle.

        :param ratio: The ratio of the obstacle's side length to
            the room's side length.
        :param tx_coords: The transmitter's coordinates, array-like, (2,).
        :param rx_coords: The receiver's coordinates, array-like, (2,).
        :return: The scene.

        :Examples:

        >>> from differt2d.scene import Scene
        >>>
        >>> scene = Scene.square_scene_with_wall()
        >>> scene.bounding_box()
        Array([[0., 0.],
               [1., 1.]], dtype=float32)
        >>> len(scene.objects)
        5
        >>> scene.transmitters["tx"]
        Point(point=Array([0.2, 0.5], dtype=float32))

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene_with_wall()
            _ = scene.plot(ax)
            plt.show()
        """
        scene = Scene.square_scene(tx_coords=tx_coords, rx_coords=rx_coords)

        wall: Object = Wall(points=jnp.array([[0.5, 0.2], [0.5, 0.8]]))

        scene.add_objects([wall])

        return scene

    @classmethod
    def square_scene_with_obstacle(cls, ratio: float = 0.1, **kwargs: Any) -> "Scene":
        """
        Instantiates a square scene with one main room, and one square obstacle in the
        center.

        :param ratio: The ratio of the obstacle's side length to
            the room's side length.
        :param kwargs:
            Keyword arguments to be passed to :meth:`square_scene`.
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
        >>> scene.transmitters["tx"]
        Point(point=Array([0.2, 0.2], dtype=float32))

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            plt.show()
        """
        scene = Scene.square_scene(**kwargs)

        hl = 0.5 * ratio

        x0, x1 = 0.5 - hl, 0.5 + hl
        y0, y1 = 0.5 - hl, 0.5 + hl

        walls: List[Object] = [
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
        transmitters: bool = True,
        transmitters_args: Sequence = (),
        transmitters_kwargs: Dict[str, Any] = {},
        objects: bool = True,
        objects_args: Sequence = (),
        objects_kwargs: Dict[str, Any] = {},
        receivers: bool = True,
        receivers_args: Sequence = (),
        receivers_kwargs: Dict[str, Any] = {},
        annotate: bool = True,
        **kwargs: Any,
    ) -> List[Artist]:
        """
        :param transmitters:
            If set, includes transmitters in the plot.
        :param transmitters_args:
            Arguments to be passed to each transmitter's plot function.
        :param transmitters_kwargs:
            Keyword arguments to be passed to each transmitter's plot function.
        :param objects:
            If set, includes objects in the plot.
        :param objects_args:
            Arguments to be passed to the each object' plot function.
        :param objects_kwargs:
            Keyword arguments to be passed to each object' plot function.
        :param receivers:
            If set, includes receivers in the plot.
        :param receivers_args:
            Arguments to be passed to each receiver's plot function.
        :param receivers_kwargs:
            Keyword arguments to be passed to each receiver's plot function.
        :param annotate:
            If set, will annotate all transmitters and receivers with their name,
            and append the corresponding artists
            to the returned list.
        """
        transmitters_kwargs.setdefault("color", "blue")
        receivers_kwargs.setdefault("color", "green")

        artists = []

        if transmitters:
            for tx_key, transmitter in self.transmitters.items():
                artists.extend(
                    transmitter.plot(
                        ax,
                        *transmitters_args,
                        *args,
                        annotate=tx_key if annotate else None,
                        **transmitters_kwargs,
                        **kwargs,
                    )
                )

        if objects:
            for obj in self.objects:
                artists.extend(
                    obj.plot(ax, *objects_args, *args, **objects_kwargs, **kwargs)  # type: ignore[union-attr]
                )

        if receivers:
            for rx_key, receiver in self.receivers.items():
                artists.extend(
                    receiver.plot(
                        ax,
                        *receivers_args,
                        *args,
                        annotate=rx_key if annotate else None,
                        **receivers_kwargs,
                        **kwargs,
                    )
                )

        return artists

    def bounding_box(self) -> Array:
        bounding_boxes_list = (
            [transmitter.bounding_box() for transmitter in self.transmitters.values()]
            + [receiver.bounding_box() for receiver in self.receivers.values()]
            + [obj.bounding_box() for obj in self.objects]  # type: ignore[union-attr]
        )
        bounding_boxes = jnp.dstack(bounding_boxes_list)

        return jnp.vstack(
            [
                jnp.min(bounding_boxes[0, :, :], axis=1),
                jnp.max(bounding_boxes[1, :, :], axis=1),
            ]
        )

    def get_closest_transmitter(self, coords: Array) -> Tuple[Point, Array]:
        """
        Returns the closest transmitter to the given coordinates.

        :param coords: The x-y coordinates, (2,).
        :return: The closet transmitter and its distance to the coordinates.
        """
        transmitters = list(self.transmitters.values())
        points = jnp.vstack([tx.point for tx in transmitters])
        i_min, distance = closest_point(points, coords)
        return transmitters[i_min], distance

    def get_closest_receiver(self, coords: Array) -> Tuple[Point, Array]:
        """
        Returns the closest receivers to the given coordinates.

        :param coords: The x-y coordinates, (2,).
        :return: The closet receiver and its distance to the coordinates.
        """
        receivers = list(self.receivers.values())
        points = jnp.vstack([rx.point for rx in receivers])
        i_min, distance = closest_point(points, coords)
        return receivers[i_min], distance

    def all_transmitter_receiver_pairs(
        self,
    ) -> Iterator[Tuple[Tuple[str, Point], Tuple[str, Point]]]:
        """
        Returns all possible pairs of (transmitter, receiver) in the scene.

        Each pair ``P := ((KE, E), (KR, R))`` is made of the following:

        + ``KE`` the name of the transmitter (key);
        + ``E`` the actual transmitter :class:`Point<differt2d.geometry.Point>` (value);
        + ``KR`` the name of the receiver (key);
        + ``R`` the actual receiver :class:`Point<differt2d.geometry.Point>` (value).

        :return: A generator of all possible pairs.
        """
        return product(self.transmitters.items(), self.receivers.items())

    @partial(jax.jit, static_argnames=["min_order", "max_order"])
    def all_path_candidates(
        self, min_order: int = 0, max_order: int = 1
    ) -> List[Array]:
        """
        Returns all path candidates, from any of the :attr:`transmitters` to any of the
        :attr:`receivers`, as a list of array of indices.

        Note that it only inclides indices for objects.

        :param min_order: The minimum order of the path, i.e., the number of
            interactions.
        :param max_order: The maximum order of the path, i.e., the number of
            interactions.
        :return: The list of list of indices.
        """
        num_primitives = len(self.objects)

        return [
            path_candidate
            for order in range(min_order, max_order + 1)
            for path_candidate in jnp.asarray(
                differt_core.generate_path_candidates(num_primitives, order),
                dtype=jnp.uint32,
            ).T
        ]

    def get_interacting_objects(self, path_candidate: Array) -> List[Interactable]:
        """
        Returns the list of interacting objects from a path candidate.

        An `interacting` object is simply an object on which the
        path should pass.

        :param path_candidates: A path candidate,
            as returned by :meth:`all_path_candidates`.
        :return: The list of interacting objects.
        """
        return [self.objects[i] for i in path_candidate]

    def all_paths(
        self,
        path_cls: Type[Path] = ImagePath,
        min_order: int = 0,
        max_order: int = 1,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, str, Array, Path, List[int]]]:
        """
        Returns all paths from any of the :attr:`transmitters` to any of the
        :attr:`receivers`, using the given method, see,
        :class:`differt2d.geometry.ImagePath` :class:`differt2d.geometry.FermatPath` and
        :class:`differt2d.geometry.MinPath`.

        :param path_cls: Method to be used to find the path coordinates.
        :param min_order:
            The minimum order of the path, i.e., the number of interactions.
        :param max_order:
            The maximum order of the path, i.e., the number of interaction
        :param kwargs:
            Keyword arguments to be passed to
            :meth:`Path.is_valid<differt2d.geometry.Path.is_valid>`.
        :return: The generator of paths, as
            (transmitter name, receiver name, valid, path, path_candidate) tuples,
            where validity path validity can be later evaluated using
            :python:`is_true(valid)`.
        """
        path_candidates = self.all_path_candidates(
            min_order=min_order, max_order=max_order
        )

        for (tx_key, transmitter), (
            rx_key,
            receiver,
        ) in self.all_transmitter_receiver_pairs():
            for path_candidate in path_candidates:
                interacting_objects = self.get_interacting_objects(path_candidate)
                path = path_cls.from_tx_objects_rx(
                    transmitter.point, interacting_objects, receiver.point
                )
                valid = path.is_valid(
                    self.objects,  # type: ignore[arg-type]
                    path_candidate,
                    interacting_objects,  # type: ignore[arg-type]
                    **kwargs,
                )

                yield (tx_key, rx_key, valid, path, path_candidate)

    def all_valid_paths(
        self,
        approx=None,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, str, Path, List[int]]]:
        """
        Returns only valid paths as returned by :meth:`all_paths`, by filtering out
        paths using :func:`is_true<differt2d.logic.is_true>`.

        :param kwargs:
            Keyword arguments to be passed to :meth:`all_paths`.
        :return: The generator of valid paths, as
            (transmitter name, receiver name, path, path_candidate) tuples.
        """
        for tx_key, rx_key, valid, path, path_candidate in self.all_paths(
            approx=approx, **kwargs
        ):
            if is_true(valid, approx=approx):
                yield (tx_key, rx_key, path, path_candidate)

    def accumulate_over_paths(
        self,
        fun: PathFun,
        fun_args: Tuple = (),
        fun_kwargs: Mapping = {},
        reduce_all: bool = False,
        **kwargs: Any,
    ) -> Iterator[Tuple[str, str, Array]]:
        """
        Repeatedly calls ``fun`` on all paths between each pair of (transmitter,
        receiver) in the scene, and accumulates the results.

        Produces an iterator with each (transmitter, receiver) pair.

        :param fun: The function to evaluate on each path.
        :param fun_args:
            Positional arguments to be passed to ``fun``.
        :param fun_kwargs:
            Keyword arguments to be passed to ``fun``.
        :param reduce_all: Whether to reduce the output by summing
            all accumulated results. This is especially useful
            if you only care about the total accumulated results.
        :param kwargs:
            Keyword arguments to be passed to
            :meth:`all_paths`.
        :return:
            An iterator of transmitter name, receiver name
            and the corresponding accumulated result, or the
            sum of accumulated results if :python:`reduce_all=True`.
        """

        def results() -> Iterator[Tuple[str, str, Array]]:
            for (tx_key, rx_key), paths_group in groupby(
                self.all_paths(**kwargs), lambda key: key[:2]
            ):
                acc = 0.0
                transmitter = self.transmitters[tx_key]
                receiver = self.receivers[rx_key]

                for _, _, valid, path, path_candidate in paths_group:
                    interacting_objects = self.get_interacting_objects(path_candidate)
                    acc = acc + valid * fun(
                        transmitter,
                        receiver,
                        path,
                        interacting_objects,  # type: ignore[arg-type]
                        *fun_args,
                        **fun_kwargs,
                    )

                yield tx_key, rx_key, acc

        if reduce_all:
            return sum(p for _, _, p in results())
        else:
            return results()

    def accumulate_on_transmitters_grid_over_paths(
        self,
        X: Array,
        Y: Array,
        fun: PathFun,
        fun_args: Tuple = (),
        fun_kwargs: Mapping = {},
        reduce_all: bool = False,
        grad: bool = False,
        value_and_grad: bool = False,
        path_cls: Type[Path] = ImagePath,
        transmitter_cls: Type[Point] = Point,
        min_order: int = 0,
        max_order: int = 1,
        **kwargs,
    ) -> Union[
        Iterator[Tuple[str, Union[Array, Tuple[Array, Array]]]],
        Union[Array, Tuple[Array, Array]],
    ]:
        """
        Repeatedly calls ``fun`` on all paths between the receivers in the scene and
        every transmitter coordinate in :python:`(X, Y)`, and accumulates the results
        over one array that has the same shape a ``X`` and ``Y``.

        Produces an iterator with one element for each receiver location.

        :param X: The grid of x-coordinates, (m, n).
        :param Y: The grid of y-coordinates, (m, n).
        :param fun: The function to evaluate on each path.
        :param fun_args:
            Positional arguments to be passed to ``fun``.
        :param fun_kwargs:
            Keyword arguments to be passed to ``fun``.
        :param reduce_all: Whether to reduce the output by summing
            all accumulated results. This is especially useful
            if you only care about the total accumulated results.
        :param grad: If set, returns the gradient of ``fun`` with respect
            to the transmitter's position. The output array(s) will then have
            an additional axis of size two.
        :param value_and_grad: If set, returns both the ``fun`` and its
            gradient. Takes precedence over setting :python:`grad=True`.
        :param path_cls: Method to be used to find the path coordinates.
        :param transmitter_cls: A point constructor called on every transmitter,
            should inherit from :class:`Point<differt2d.geometry.Point>`.
        :param min_order:
            The minimum order of the path, i.e., the number of interactions.
        :param max_order:
            The maximum order of the path, i.e., the number of interactions.
        :param kwargs:
            Keyword arguments to be passed to
            :meth:`Path.is_valid<differt2d.geometry.Path.is_valid>`.
        :return:
            An iterator of receiver name and the corresponding
            accumulated result, or the sum of accumulated results
            if :python:`reduce_all=True`.
        """
        transmitters = self.transmitters
        self.transmitters = {"tx": Point(point=jnp.array([0.0, 0.0]))}

        path_candidates = self.all_path_candidates(
            min_order=min_order,
            max_order=max_order,
        )

        pairs = list(self.all_transmitter_receiver_pairs())
        self.transmitters = transmitters

        def facc(tx_coords: Array, receiver: Point) -> Array:
            acc = 0.0
            for path_candidate in path_candidates:
                interacting_objects = self.get_interacting_objects(path_candidate)
                path = path_cls.from_tx_objects_rx(
                    tx_coords, interacting_objects, receiver.point
                )
                valid = path.is_valid(
                    self.objects,  # type: ignore[arg-type]
                    path_candidate,
                    interacting_objects,  # type: ignore[arg-type]
                    **kwargs,
                )
                acc = acc + valid * fun(
                    transmitter_cls(point=tx_coords),
                    receiver,
                    path,
                    interacting_objects,
                    *fun_args,
                    **fun_kwargs,
                )

            return acc

        if value_and_grad:
            facc = jax.value_and_grad(facc, argnums=0)
        elif grad:
            facc = jax.grad(facc, argnums=0)

        vfacc = jax.vmap(
            jax.vmap(facc, in_axes=(0, None)),
            in_axes=(0, None),
        )

        grid = jnp.dstack((X, Y))

        def results() -> Iterator[Array]:
            return ((rx_key, vfacc(grid, receiver)) for _, (rx_key, receiver) in pairs)

        if reduce_all:
            if value_and_grad:
                Z = 0.0
                dZ = 0.0
                for _, (p, dp) in results():
                    Z = Z + p
                    dZ = dZ + dp

                return Z, dZ
            else:
                return sum(p for _, p in results())
        else:
            return results()

    def accumulate_on_receivers_grid_over_paths(
        self,
        X: Array,
        Y: Array,
        fun: PathFun,
        fun_args: Tuple = (),
        fun_kwargs: Mapping = {},
        reduce_all: bool = False,
        grad: bool = False,
        value_and_grad: bool = False,
        path_cls: Type[Path] = ImagePath,
        receiver_cls: Type[Point] = Point,
        min_order: int = 0,
        max_order: int = 1,
        **kwargs,
    ) -> Union[
        Iterator[Tuple[str, Union[Array, Tuple[Array, Array]]]],
        Union[Array, Tuple[Array, Array]],
    ]:
        """
        Repeatedly calls ``fun`` on all paths between the transmitters in the scene and
        every receiver coordinate in :python:`(X, Y)`, and accumulates the results over
        one array that has the same shape a ``X`` and ``Y``.

        Produces an iterator with one element for each transmitter location.

        :param X: The grid of x-coordinates, (m, n).
        :param Y: The grid of y-coordinates, (m, n).
        :param fun: The function to evaluate on each path.
        :param fun_args:
            Positional arguments to be passed to ``fun``.
        :param fun_kwargs:
            Keyword arguments to be passed to ``fun``.
        :param reduce_all: Whether to reduce the output by summing
            all accumulated results. This is especially useful
            if you only care about the total accumulated results.
        :param grad: If set, returns the gradient of ``fun`` with respect
            to the receiver's position. The output array(s) will then have
            an additional axis of size two.
        :param value_and_grad: If set, returns both the ``fun`` and its
            gradient. Takes precedence over setting :python:`grad=True`.
        :param path_cls: Method to be used to find the path coordinates.
        :param receiver_cls: A point constructor called on every receiver,
            should inherit from :class:`Point<differt2d.geometry.Point>`.
        :param min_order:
            The minimum order of the path, i.e., the number of interactions.
        :param max_order:
            The maximum order of the path, i.e., the number of interactions.
        :param kwargs:
            Keyword arguments to be passed to
            :meth:`Path.is_valid<differt2d.geometry.Path.is_valid>`.
        :return:
            An iterator of transmitter name and the corresponding
            accumulated result, or the sum of accumulated results
            if :python:`reduce_all=True`.
        """
        receivers = self.receivers
        self.receivers = {"rx": Point(point=jnp.array([0.0, 0.0]))}

        path_candidates = self.all_path_candidates(
            min_order=min_order,
            max_order=max_order,
        )

        pairs = list(self.all_transmitter_receiver_pairs())
        self.receivers = receivers

        def facc(transmitter: Point, rx_coords: Array) -> Array:
            acc = 0.0
            for path_candidate in path_candidates:
                interacting_objects = self.get_interacting_objects(path_candidate)
                path = path_cls.from_tx_objects_rx(
                    transmitter.point, interacting_objects, rx_coords
                )
                valid = path.is_valid(
                    self.objects,  # type: ignore[arg-type]
                    path_candidate,
                    interacting_objects,  # type: ignore[arg-type]
                    **kwargs,
                )
                acc = acc + valid * fun(
                    transmitter,
                    receiver_cls(point=rx_coords),
                    path,
                    interacting_objects,
                    *fun_args,
                    **fun_kwargs,
                )

            return acc

        if value_and_grad:
            facc = jax.value_and_grad(facc, argnums=1)
        elif grad:
            facc = jax.grad(facc, argnums=1)

        vfacc = jax.vmap(
            jax.vmap(facc, in_axes=(None, 0)),
            in_axes=(None, 0),
        )

        grid = jnp.dstack((X, Y))

        def results() -> Iterator[Array]:
            return (
                (tx_key, vfacc(transmitter, grid)) for (tx_key, transmitter), _ in pairs
            )

        if reduce_all:
            if value_and_grad:
                Z = 0.0
                dZ = 0.0
                for _, (p, dp) in results():
                    Z = Z + p
                    dZ = dZ + dp

                return Z, dZ
            else:
                return sum(p for _, p in results())
        else:
            return results()
