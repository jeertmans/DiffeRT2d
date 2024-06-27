"""Scenes for tracing rays between transmitters and receivers."""

__all__ = ("Scene", "SceneName")

import json
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import singledispatchmethod
from itertools import groupby, product
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from differt_core.rt.graph import CompleteGraph, DiGraph
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from matplotlib.artist import Artist

from ._typing import ScalarFloat
from .abc import Interactable, Loc, Object, Plottable
from .geometry import ImagePath, Path, Point, Wall, closest_point
from .logic import Truthy, is_true

PathFun = Callable[[Point, Point, Path, list[Interactable]], Float[Array, " "]]

S = Union[str, bytes, bytearray]
SceneName = Literal[
    "basic_scene",
    "square_scene",
    "square_scene_with_obstacle",
    "square_scene_with_wall",
]
"""Literal type for all valid scene names."""
_K = TypeVar("_K")
_V = TypeVar("_V")
_O = TypeVar("_O", bound=Object, covariant=True)


@runtime_checkable
class Readable(Protocol):
    @abstractmethod
    def read(self) -> S: ...


class PyTreeDict(eqx.Module, Mapping[_K, _V]):
    """
    An immutable mapping that is also a PyTree.

    The main difference with the usual dict is that, here, the index
    time is linear with the size of the mapping.
    """

    _keys: tuple[_K, ...] = eqx.field(converter=lambda seq: tuple(seq), static=True)
    """The sequence of keys."""
    _values: tuple[_V, ...] = eqx.field(converter=lambda seq: tuple(seq))
    """The sequence of values."""

    def __check_init__(self):
        if len(self._keys) != len(self._values):
            raise ValueError(
                "Number of keys must match number of values, "
                f"got {len(self._keys)} and {len(self._values)}."
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[_K, _V]) -> "PyTreeDict":
        """
        Constructs an immutable mapping from another mapping.

        :param: An existing mapping.
        :return: The new mapping.
        """
        return cls(
            _keys=mapping.keys(),  # type: ignore[reportArgumentType]
            _values=mapping.values(),  # type: ignore[reportArgumentType]
        )

    def __getitem__(self, key: _K) -> _V:
        try:
            index = self._keys.index(key)
            return self._values[index]
        except ValueError as e:
            raise KeyError from e

    def __iter__(self) -> Iterator[_K]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


@eqx.filter_jit
def all_path_candidates(
    num_nodes: int,
    min_order: int = 0,
    max_order: int = 1,
    *,
    order: Optional[int] = None,
    filter_nodes: Optional[tuple[int, ...]] = None,
) -> list[Int[Array, "num_path_candidates order"]]:
    """
    Returns all path candidates, as a list of array of indices.

    This function is a lower-level version of :meth:`Scene.all_path_candidates`
    that utilizes caching for more efficient computations.

    In practice, you should not use this function directly, but rather
    :meth:`Scene.all_path_candidates`.

    :param num_nodes: The number of nodes, i.e., the objects,
        present in the `graph`. This excludes the transmitting
        and receiving nodes.
    :param min_order: The minimum order of the path, i.e., the
        number of interactions.
    :param max_order: The maximum order of the path, i.e., the
        number of interactions.
    :param order: If provided, it is equivalent to setting
        ``min_order=order`` and ``max_order=order``.
    :param filter_nodes: An optional list of nodes, i.e., object indices,
        to not visit.
    :return: The list of list of indices.
    """
    if filter_nodes is None:
        graph = CompleteGraph(num_nodes)
        from_ = num_nodes
        to = from_ + 1
    else:
        graph = DiGraph.from_complete_graph(CompleteGraph(num_nodes))
        from_, to = graph.insert_from_and_to_nodes()
        graph.disconnect_nodes(*filter_nodes)

    if order is not None:
        min_order = order
        max_order = order

    return [
        jnp.asarray(path_candidate, dtype=jnp.int32)
        for order in range(min_order, max_order + 1)
        for path_candidate in graph.all_paths(
            from_, to, order + 2, include_from_and_to=False
        )
    ]


class Scene(Plottable, eqx.Module, Generic[_O]):
    """2D Scene made of objects, one or more transmitting node(s), and one or more receiving node(s)."""

    transmitters: Mapping[str, Point] = eqx.field(
        converter=lambda d: PyTreeDict.from_mapping(d),
        default_factory=lambda: PyTreeDict.from_mapping({}),
    )
    """The transmitting nodes."""
    receivers: Mapping[str, Point] = eqx.field(
        converter=lambda d: PyTreeDict.from_mapping(d),
        default_factory=lambda: PyTreeDict.from_mapping({}),
    )
    """The receiving nodes."""
    objects: Sequence[_O] = ()
    """The sequence of objects in the scene."""

    @jaxtyped(typechecker=typechecker)
    def with_transmitters(self, **transmitters: Point) -> "Scene":
        """
        Returns a copy of this scene, with the given transmitters.

        :param transmitters: A mapping of transmitter names and points.
        :return: The new scene.
        """
        return eqx.tree_at(
            lambda s: s.transmitters, self, PyTreeDict.from_mapping(transmitters)
        )

    @jaxtyped(typechecker=typechecker)
    def with_receivers(self, **receivers: Point) -> "Scene":
        """
        Returns a copy of this scene, with the given receivers.

        :param receivers: A mapping of receiver names and points.
        :return: The new scene.
        """
        return eqx.tree_at(
            lambda s: s.receivers, self, PyTreeDict.from_mapping(receivers)
        )

    @jaxtyped(typechecker=typechecker)
    def with_objects(self, *objects: Object) -> "Scene":
        """
        Returns a copy of this scene, with the given objects.

        :param objects: A sequence of objects.
        :return: The new scene.
        """
        return eqx.tree_at(lambda s: s.objects, self, tuple(objects))

    @jaxtyped(typechecker=typechecker)
    def filter_objects(self, filter_spec: Callable[[Object], bool]) -> "Scene":
        """
        Returns a copy of this scene, with the given objects filtered out.

        :param filter_spec: A callable indicating
            which objects should be kept.
        :return: The new scene.

        :Examples:

        A practical use case for this function is to plot a basic scene,
        but only perform ray tracing on a subset of the objects, e.g.,
        simulating only vertex diffraction.

        .. warning::
            If you only want to simulate vertex diffraction,
            but still take the objects into account for possible
            obstruction, prefer using the ``filter_objects``
            parameter in :meth:`all_path_candidates` and
            associated methods.

        .. plot::
            :include-source:

            import jax
            import matplotlib.pyplot as plt

            from differt2d.geometry import FermatPath, Vertex
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene_with_wall()
            wall = scene.objects[-1]
            scene = scene.add_objects(*wall.get_vertices())
            _ = scene.plot(ax)
            scene = scene.filter_objects(lambda o: isinstance(o, Vertex))
            key = jax.random.PRNGKey(1234)

            for _, _, path, _ in scene.all_valid_paths(
                order=1,
                path_cls=FermatPath,
                key=key,
            ):
                path.plot(ax)

            plt.show()  # doctest: +SKIP
        """
        return eqx.tree_at(
            lambda s: s.objects, self, tuple(filter(filter_spec, self.objects))
        )

    @jaxtyped(typechecker=typechecker)
    def update_transmitters(self, **transmitters: Point) -> "Scene":
        """
        Returns a copy of this scene, with the updated transmitters.

        The new set of transmitters is the union of the previous set and
        the ones provided as arguments.

        :param transmitters: A mapping of transmitter names and points.
        :return: The new scene.
        """
        return eqx.tree_at(
            lambda s: s.transmitters,
            self,
            PyTreeDict.from_mapping({**self.transmitters, **transmitters}),
        )

    @jaxtyped(typechecker=typechecker)
    def update_receivers(self, **receivers: Point) -> "Scene":
        """
        Returns a copy of this scene, with the updated receivers.

        The new set of receivers is the union of the previous set and
        the ones provided as arguments.

        :param receivers: A mapping of receivers names and points.
        :return: The new scene.
        """
        return eqx.tree_at(
            lambda s: s.receivers,
            self,
            PyTreeDict.from_mapping({**self.receivers, **receivers}),
        )

    @jaxtyped(typechecker=typechecker)
    def add_objects(self, *objects: Object) -> "Scene":
        """
        Returns a copy of this scene, with the given objects, plus the scene objects.

        :param objects: A sequence of objects.
        :return: The new scene.
        """
        return self.with_objects(*self.objects, *objects)

    @jaxtyped(typechecker=typechecker)
    def rename_transmitters(self, **transmitter_names: str) -> "Scene":
        """
        Returns a copy of this scene, with the specified transmitters renamed.

        :param transmitter_names: A mapping of transmitter old names
            and new names.
        :return: The new scene.
        """
        transmitters = {}
        for name, point in self.transmitters.items():
            name = transmitter_names.get(name, name)
            transmitters[name] = point

        return self.with_transmitters(**transmitters)

    @jaxtyped(typechecker=typechecker)
    def rename_receivers(self, **receiver_names: str) -> "Scene":
        """
        Returns a copy of this scene, with the specified receivers renamed.

        :param receiver_names: A mapping of receiver old names
            and new names.
        :return: The new scene.
        """
        receivers = {}
        for name, point in self.receivers.items():
            name = receiver_names.get(name, name)
            receivers[name] = point

        return self.with_receivers(**receivers)

    @classmethod
    @jaxtyped(typechecker=typechecker)
    def from_walls_array(cls, walls: Float[Array, "num_walls 2 2"]) -> "Scene":
        """
        Creates an empty scene from an array of walls.

        :param walls: An array of wall coordinates.
        :return: The new scene.
        """
        return cls(
            transmitters={},  # type: ignore[reportArgumentType]
            receivers={},  # type: ignore[reportArgumentType]
            objects=[Wall(xys=xys) for xys in walls],
        )

    @singledispatchmethod
    @classmethod
    def from_geojson(
        cls, s_or_fp: Union[S, Readable], tx_loc: Loc = "NW", rx_loc: Loc = "SE"
    ) -> "Scene":
        r"""
        Creates a scene from a GEOJSON file, generating one Wall per line segment. TX and RX positions are located on the corner of the bounding box.

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
            :include-source: false

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
                        xys = jnp.array(
                            [coordinates[i - 1], coordinates[i]], dtype=float
                        )
                        wall = Wall(xys=xys)
                        walls.append(wall)

        scene = Scene(objects=walls)

        if len(walls) > 0:
            scene = scene.with_transmitters(tx=Point(xy=scene.get_location(tx_loc)))
            scene = scene.with_receivers(rx=Point(xy=scene.get_location(rx_loc)))
        else:
            scene = scene.with_transmitters(tx=Point(xy=jnp.array([0.0, 0.0])))
            scene = scene.with_receivers(rx=Point(xy=jnp.array([1.0, 1.0])))

        return scene

    @from_geojson.register(Readable)
    @classmethod
    def _(cls, fp: Readable, *args: Any, **kwargs: Any) -> "Scene":
        return cls.from_geojson(fp.read(), *args, **kwargs)

    @classmethod
    def from_scene_name(
        cls, scene_name: SceneName, *args: Any, **kwargs: Any
    ) -> "Scene":
        """
        Generates a new scene from the given scene name.

        :param scene_name: The name of the scene.
        :param args: Positional arguments passed to the
            constructor.
        :param kwargs: Keyword arguments passed to the
            constructor.
        :return: The scene.
        """
        return getattr(cls, scene_name)(*args, **kwargs)

    @classmethod
    def random_uniform_scene(
        cls,
        n_transmitters: int = 1,
        n_walls: int = 1,
        n_receivers: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> "Scene":
        """
        Generates a random scene, drawing coordinates from a random distribution.

        :param key: The random key to be used.
        :param n_transmitters: The number of transmitters.
        :param n_walls: The number of walls.
        :param n_receivers: The number of receivers.
        :return: The scene.

        :Examples:

        .. plot::
            :include-source:

            import matplotlib.pyplot as plt
            import jax
            from differt2d.scene import Scene

            ax = plt.gca()
            key = jax.random.PRNGKey(1234)
            scene = Scene.random_uniform_scene(n_walls=5, key=key)
            _ = scene.plot(ax)
            plt.show()  # doctest: +SKIP

        """
        points = jax.random.uniform(
            key, (n_transmitters + 2 * n_walls + n_receivers, 2)
        )
        transmitters = {
            f"tx_{i}": Point(xy=points[i, :]) for i in range(n_transmitters)
        }
        receivers = {
            f"rx_{i}": Point(xy=points[-(i + 1), :]) for i in range(n_receivers)
        }

        walls = [
            Wall(xys=points[2 * i + n_transmitters : 2 * i + 2 + n_transmitters, :])
            for i in range(n_walls)
        ]
        return cls(transmitters=transmitters, receivers=receivers, objects=walls)

    @classmethod
    def basic_scene(
        cls,
        tx_coords: Float[Array, "2"] = jnp.array([0.1, 0.1]),  # noqa: B008
        rx_coords: Float[Array, "2"] = jnp.array([0.302, 0.2147]),  # noqa: B008
    ) -> "Scene":
        """
        Instantiates a basic scene with a main room, and a second inner room in the lower left corner, with a small entrance.

        :param tx_coords: The transmitter's coordinates.
        :param rx_coords: The receiver's coordinates.
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
        >>> scene.transmitters["tx"].xy
        Array([0.1, 0.1], dtype=float32)

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.basic_scene()
            _ = scene.plot(ax)
            plt.show()  # doctest: +SKIP

        """
        tx = Point(xy=jnp.asarray(tx_coords, dtype=float))
        rx = Point(xy=jnp.asarray(rx_coords, dtype=float))

        walls = [
            # Outer walls
            Wall(xys=jnp.array([[0.0, 0.0], [1.0, 0.0]])),
            Wall(xys=jnp.array([[1.0, 0.0], [1.0, 1.0]])),
            Wall(xys=jnp.array([[1.0, 1.0], [0.0, 1.0]])),
            Wall(xys=jnp.array([[0.0, 1.0], [0.0, 0.0]])),
            # Small room
            Wall(xys=jnp.array([[0.4, 0.0], [0.4, 0.4]])),
            Wall(xys=jnp.array([[0.4, 0.4], [0.3, 0.4]])),
            Wall(xys=jnp.array([[0.1, 0.4], [0.0, 0.4]])),
        ]

        return cls(transmitters={"tx": tx}, receivers={"rx": rx}, objects=walls)

    @classmethod
    def square_scene(
        cls,
        tx_coords: Float[Array, "2"] = jnp.array([0.2, 0.2]),  # noqa: B008
        rx_coords: Float[Array, "2"] = jnp.array([0.5, 0.6]),  # noqa: B008
    ) -> "Scene":
        """
        Instantiates a square scene with one main room.

        :param tx_coords: The transmitter's coordinates.
        :param rx_coords: The receiver's coordinates.
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
        >>> scene.transmitters["tx"].xy
        Array([0.2, 0.2], dtype=float32)

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            plt.show()  # doctest: +SKIP

        """
        tx = Point(xy=jnp.asarray(tx_coords, dtype=float))
        rx = Point(xy=jnp.asarray(rx_coords, dtype=float))

        walls = [
            Wall(xys=jnp.array([[0.0, 0.0], [1.0, 0.0]])),
            Wall(xys=jnp.array([[1.0, 0.0], [1.0, 1.0]])),
            Wall(xys=jnp.array([[1.0, 1.0], [0.0, 1.0]])),
            Wall(xys=jnp.array([[0.0, 1.0], [0.0, 0.0]])),
        ]

        return cls(transmitters={"tx": tx}, receivers={"rx": rx}, objects=walls)

    @classmethod
    def square_scene_with_wall(
        cls,
        ratio: float = 0.1,
        tx_coords: Float[Array, "2"] = jnp.array([0.2, 0.5]),  # noqa: B008
        rx_coords: Float[Array, "2"] = jnp.array([0.8, 0.5]),  # noqa: B008
    ) -> "Scene":
        """
        Instantiates a square scene with one main room, and vertical wall in the middle.

        :param ratio: The ratio of the obstacle's side length to
            the room's side length.
        :param tx_coords: The transmitter's coordinates.
        :param rx_coords: The receiver's coordinates.
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
        >>> scene.transmitters["tx"].xy
        Array([0.2, 0.5], dtype=float32)

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene_with_wall()
            _ = scene.plot(ax)
            plt.show()  # doctest: +SKIP

        """
        scene = Scene.square_scene(tx_coords=tx_coords, rx_coords=rx_coords)

        scene = scene.add_objects(Wall(xys=jnp.array([[0.5, 0.2], [0.5, 0.8]])))

        return scene

    @classmethod
    def square_scene_with_obstacle(
        cls, ratio: ScalarFloat = 0.1, **kwargs: Any
    ) -> "Scene":
        """
        Instantiates a square scene with one main room, and one square obstacle in the center.

        :param ratio: The ratio of the obstacle's side length to
            the room's side length.
        :param kwargs:
            Keyword arguments passed to :meth:`square_scene`.
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
        >>> scene.transmitters["tx"].xy
        Array([0.2, 0.2], dtype=float32)

        .. plot::

            import matplotlib.pyplot as plt
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            plt.show()  # doctest: +SKIP

        """
        scene = Scene.square_scene(**kwargs)

        hl = 0.5 * ratio

        x0, x1 = 0.5 - hl, 0.5 + hl
        y0, y1 = 0.5 - hl, 0.5 + hl

        scene = scene.add_objects(
            Wall(xys=jnp.array([[x0, y0], [x1, y0]])),
            Wall(xys=jnp.array([[x1, y0], [x1, y1]])),
            Wall(xys=jnp.array([[x1, y1], [x0, y1]])),
            Wall(xys=jnp.array([[x0, y1], [x0, y0]])),
        )

        return scene

    def plot(
        self,
        ax,
        *args: Any,
        transmitters: bool = True,
        transmitters_args: tuple = (),
        transmitters_kwargs: Optional[Mapping[str, Any]] = None,
        objects: bool = True,
        objects_args: tuple = (),
        objects_kwargs: Optional[Mapping[str, Any]] = None,
        receivers: bool = True,
        receivers_args: tuple = (),
        receivers_kwargs: Optional[Mapping[str, Any]] = None,
        annotate: bool = True,
        **kwargs: Any,
    ) -> list[Artist]:
        """
        :param transmitters:
            If set, includes transmitters in the plot.
        :param transmitters_args:
            Arguments passed to each transmitter's plot function.
        :param transmitters_kwargs:
            Keyword arguments passed to each transmitter's plot function.
        :param objects:
            If set, includes objects in the plot.
        :param objects_args:
            Arguments passed to the each object' plot function.
        :param objects_kwargs:
            Keyword arguments passed to each object' plot function.
        :param receivers:
            If set, includes receivers in the plot.
        :param receivers_args:
            Arguments passed to each receiver's plot function.
        :param receivers_kwargs:
            Keyword arguments passed to each receiver's plot function.
        :param annotate:
            If set, will annotate all transmitters and receivers with their name,
            and append the corresponding artists
            to the returned list.
        """  # noqa: D205
        if receivers_kwargs is None:
            receivers_kwargs = {}
        if objects_kwargs is None:
            objects_kwargs = {}
        if transmitters_kwargs is None:
            transmitters_kwargs = {}

        transmitters_kwargs = {"color": "blue", **transmitters_kwargs}
        receivers_kwargs = {"color": "green", **receivers_kwargs}

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

    def bounding_box(self) -> Float[Array, "2 2"]:  # noqa: D102
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

    @jaxtyped(typechecker=typechecker)
    def get_closest_transmitter(
        self, coords: Float[Array, "2"]
    ) -> tuple[str, Float[Array, " "]]:
        """
        Returns the closest transmitter to the given coordinates.

        :param coords: The x-y coordinates.
        :return: The closet transmitter (name) and its distance to the
            coordinates.
        """
        transmitters = list(self.transmitters.items())
        points = jnp.vstack([tx.xy for _, tx in transmitters])
        i_min, distance = closest_point(points, coords)
        return transmitters[i_min][0], distance

    @jaxtyped(typechecker=typechecker)
    def get_closest_receiver(
        self, coords: Float[Array, "2"]
    ) -> tuple[str, Float[Array, " "]]:
        """
        Returns the closest receivers to the given coordinates.

        :param coords: The x-y coordinates.
        :return: The closet receiver (name) and its distance to the
            coordinates.
        """
        receivers = list(self.receivers.items())
        points = jnp.vstack([rx.xy for _, rx in receivers])
        i_min, distance = closest_point(points, coords)
        return receivers[i_min][0], distance

    def all_transmitter_receiver_pairs(
        self,
    ) -> Iterator[tuple[tuple[str, Point], tuple[str, Point]]]:
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

    @eqx.filter_jit
    def all_path_candidates(
        self,
        min_order: int = 0,
        max_order: int = 1,
        *,
        order: Optional[int] = None,
        filter_objects: Optional[Callable[[Object], bool]] = None,
    ) -> list[Int[Array, "num_path_candidates order"]]:
        """
        Returns all path candidates, from any of the :attr:`transmitters` to any of the :attr:`receivers`, as a list of array of indices.

        Note that it only includes indices for objects.

        .. note::

            Internally, it uses :py:class:`differt_core.rt.graph.CompleteGraph`
            to generate the sequence of all path candidates efficiently.

        :param min_order: The minimum order of the path, i.e., the
            number of interactions.
        :param max_order: The maximum order of the path, i.e., the
            number of interactions.
        :param order: If provided, it is equivalent to setting
            ``min_order=order`` and ``max_order=order``.
        :param filter_objects: A callable indicating
            which objects should be used for path candidates.
        :return: The list of list of indices.
        """
        num_nodes = len(self.objects)

        if filter_objects is None:
            filter_nodes = None
        else:
            filter_nodes = ()
            for i, obj in enumerate(self.objects):
                if not filter_objects(obj):
                    filter_nodes = (*filter_nodes, i)

        return all_path_candidates(
            num_nodes,
            min_order=min_order,
            max_order=max_order,
            order=order,
            filter_nodes=filter_nodes,
        )

    def get_interacting_objects(
        self, path_candidate: Int[Array, " order"]
    ) -> list[Interactable]:
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
        path_cls: type[Path] = ImagePath,
        path_cls_kwargs: Optional[Mapping[str, Any]] = None,
        min_order: int = 0,
        max_order: int = 1,
        order: Optional[int] = None,
        filter_objects: Optional[Callable[[Object], bool]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs: Any,
    ) -> Iterator[tuple[str, str, Truthy, Path, Int[Array, " order"]]]:
        """
        Returns all paths from any of the :attr:`transmitters` to any of the :attr:`receivers`, using the given method, see, :class:`differt2d.geometry.ImagePath` :class:`differt2d.geometry.FermatPath` and :class:`differt2d.geometry.MinPath`.

        :param path_cls: Method to be used to find the path coordinates.
        :param path_cls_kwargs:
            Keyword arguments passed to ``path_cls.from_tx_objects_rx``.
        :param min_order: The minimum order of the path, i.e., the
            number of interactions.
        :param max_order: The maximum order of the path, i.e., the
            number of interactions.
        :param order: If provided, it is equivalent to setting
            ``min_order=order`` and ``max_order=order``.
        :param filter_objects: A callable indicating
            which objects should be used for path candidates.
        :param key: The random key to be used to find the paths.
            Depending on ``path_cls``, this can be mandatory.
        :param kwargs:
            Keyword arguments passed to
            :meth:`Path.is_valid<differt2d.geometry.Path.is_valid>`.
        :return: The generator of paths, as
            (transmitter name, receiver name, valid, path, path_candidate) tuples,
            where validity path validity can be later evaluated using
            :python:`is_true(valid)`.
        """
        if path_cls_kwargs is None:  # pragma: no cover  # TODO: add test
            path_cls_kwargs = {}

        path_candidates = self.all_path_candidates(
            min_order=min_order,
            max_order=max_order,
            order=order,
            filter_objects=filter_objects,
        )

        for (tx_key, transmitter), (
            rx_key,
            receiver,
        ) in self.all_transmitter_receiver_pairs():
            for path_candidate in path_candidates:
                interacting_objects = self.get_interacting_objects(path_candidate)

                if key is not None:
                    key, key_path = jax.random.split(key, 2)
                else:
                    key_path = None

                path = path_cls.from_tx_objects_rx(
                    transmitter,
                    interacting_objects,
                    receiver,
                    key=key_path,
                    **path_cls_kwargs,
                )
                valid = path.is_valid(
                    self.objects,
                    path_candidate,
                    interacting_objects,
                    **kwargs,
                )

                yield (tx_key, rx_key, valid, path, path_candidate)

    def all_valid_paths(
        self,
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[tuple[str, str, Path, Int[Array, " order"]]]:
        """
        Returns only valid paths as returned by :meth:`all_paths`, by filtering out paths using :func:`is_true<differt2d.logic.is_true>`.

        :param kwargs:
            Keyword arguments passed to :meth:`all_paths`.
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
        fun_args: tuple = (),
        fun_kwargs: Optional[Mapping[str, Any]] = None,
        reduce_all: bool = False,
        **kwargs: Any,
    ) -> Union[Iterator[tuple[str, str, Float[Array, " "]]], Float[Array, " "]]:
        """
        Repeatedly calls ``fun`` on all paths between each pair of (transmitter, receiver) in the scene, and accumulates the results.

        Produces an iterator with each (transmitter, receiver) pair.

        :param fun: The function to evaluate on each path.
        :param fun_args:
            Positional arguments passed to ``fun``.
        :param fun_kwargs:
            Keyword arguments passed to ``fun``.
        :param reduce_all: Whether to reduce the output by summing
            all accumulated results. This is especially useful
            if you only care about the total accumulated results.
        :param kwargs:
            Keyword arguments passed to
            :meth:`all_paths`.
        :return:
            An iterator of transmitter name, receiver name
            and the corresponding accumulated result, or the
            sum of accumulated results if :python:`reduce_all=True`.
        """
        if fun_kwargs is None:
            fun_kwargs = {}

        def results() -> Iterator[tuple[str, str, Float[Array, " "]]]:
            for (tx_key, rx_key), paths_group in groupby(
                self.all_paths(**kwargs), lambda key: key[:2]
            ):
                acc = jnp.array(0.0)
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
            Z = jnp.array(0.0)

            for _, _, p in results():
                Z = Z + p

            return Z
        else:
            return results()

    def accumulate_on_transmitters_grid_over_paths(  # noqa: C901
        self,
        X: Float[Array, "m n"],
        Y: Float[Array, "m n"],
        fun: PathFun,
        fun_args: tuple = (),
        fun_kwargs: Optional[Mapping[str, Any]] = None,
        reduce_all: bool = False,
        grad: bool = False,
        value_and_grad: bool = False,
        path_cls: type[Path] = ImagePath,
        path_cls_kwargs: Optional[Mapping[str, Any]] = None,
        transmitter_cls: type[Point] = Point,
        min_order: int = 0,
        max_order: int = 1,
        order: Optional[int] = None,
        filter_objects: Optional[Callable[[Object], bool]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Union[
        Iterator[
            tuple[
                str,
                Union[
                    Union[Float[Array, "m n"], Float[Array, "m n 2"]],
                    tuple[Float[Array, "m n"], Float[Array, "m n 2"]],
                ],
            ]
        ],
        Union[
            Union[Float[Array, "m n"], Float[Array, "m n 2"]],
            tuple[Float[Array, "m n"], Float[Array, "m n 2"]],
        ],
    ]:
        """
        Repeatedly calls ``fun`` on all paths between the receivers in the scene and every transmitter coordinate in :python:`(X, Y)`, and accumulates the results over one array that has the same shape a ``X`` and ``Y``.

        Produces an iterator with one element for each receiver location.

        :param X: The grid of x-coordinates.
        :param Y: The grid of y-coordinates.
        :param fun: The function to evaluate on each path.
        :param fun_args:
            Positional arguments passed to ``fun``.
        :param fun_kwargs:
            Keyword arguments passed to ``fun``.
        :param reduce_all: Whether to reduce the output by summing
            all accumulated results. This is especially useful
            if you only care about the total accumulated results.
        :param grad: If set, returns the gradient of ``fun`` with respect
            to the transmitter's position. The output array(s) will then have
            an additional axis of size two.
        :param value_and_grad: If set, returns both the ``fun`` and its
            gradient. Takes precedence over setting :python:`grad=True`.
        :param path_cls: Method to be used to find the path coordinates.
        :param path_cls_kwargs:
            Keyword arguments passed to ``path_cls.from_tx_objects_rx``.
        :param transmitter_cls: A point constructor called on every transmitter,
            should inherit from :class:`Point<differt2d.geometry.Point>`.
        :param min_order:
            The minimum order of the path, i.e., the number of interactions.
        :param max_order:
            The maximum order of the path, i.e., the number of interactions.
        :param order: If provided, it is equivalent to setting
            ``min_order=order`` and ``max_order=order``.
        :param filter_objects: A callable indicating
            which objects should be used for path candidates.
        :param key: The random key to be used to find the paths.
            Depending on ``path_cls``, this can be mandatory.
        :param kwargs:
            Keyword arguments passed to
            :meth:`Path.is_valid<differt2d.geometry.Path.is_valid>`.
        :return:
            An iterator of receiver name and the corresponding
            accumulated result, or the sum of accumulated results
            if :python:`reduce_all=True`.
        """
        if fun_kwargs is None:
            fun_kwargs = {}

        if path_cls_kwargs is None:
            path_cls_kwargs = {}

        scene = self.with_transmitters(tx=Point(xy=jnp.array([0.0, 0.0])))

        path_candidates = scene.all_path_candidates(
            min_order=min_order,
            max_order=max_order,
            order=order,
            filter_objects=filter_objects,
        )

        pairs = list(scene.all_transmitter_receiver_pairs())

        if key is not None:
            keys = list(jax.random.split(key, len(path_candidates)))
        else:
            keys = [None] * len(path_candidates)

        def facc(tx_coords: Float[Array, "2"], receiver: Point) -> Float[Array, " "]:
            acc = jnp.array(0.0)
            for path_candidate, key in zip(path_candidates, keys):
                interacting_objects = scene.get_interacting_objects(path_candidate)
                path = path_cls.from_tx_objects_rx(
                    tx_coords,
                    interacting_objects,
                    receiver,
                    key=key,
                    **path_cls_kwargs,
                )
                valid = path.is_valid(
                    scene.objects,
                    path_candidate,
                    interacting_objects,
                    **kwargs,
                )
                acc = acc + valid * fun(
                    transmitter_cls(xy=tx_coords),
                    receiver,
                    path,
                    interacting_objects,
                    *fun_args,
                    **fun_kwargs,
                )

            return acc

        if value_and_grad:
            f = jax.value_and_grad(facc, argnums=0)  # type: ignore
        elif grad:
            f = jax.grad(facc, argnums=0)
        else:
            f = facc

        vf = jax.vmap(
            jax.vmap(f, in_axes=(0, None)),
            in_axes=(0, None),
        )

        grid = jnp.dstack((X, Y))

        def results() -> Iterator[tuple[str, Union[Array, tuple[Array, Array]]]]:
            return ((rx_key, vf(grid, receiver)) for _, (rx_key, receiver) in pairs)

        if reduce_all:
            if value_and_grad:
                Z = jnp.array(0.0)
                dZ = jnp.array(0.0)
                for _, (p, dp) in results():
                    Z = Z + p
                    dZ = dZ + dp

                return Z, dZ
            else:
                Z = jnp.array(0.0)
                for _, p in results():
                    Z = Z + p

                return Z
        else:
            return results()

    def accumulate_on_receivers_grid_over_paths(  # noqa: C901
        self,
        X: Float[Array, "m n"],
        Y: Float[Array, "m n"],
        fun: PathFun,
        fun_args: tuple = (),
        fun_kwargs: Optional[Mapping[str, Any]] = None,
        reduce_all: bool = False,
        grad: bool = False,
        value_and_grad: bool = False,
        path_cls: type[Path] = ImagePath,
        path_cls_kwargs: Optional[Mapping[str, Any]] = None,
        receiver_cls: type[Point] = Point,
        min_order: int = 0,
        max_order: int = 1,
        order: Optional[int] = None,
        filter_objects: Optional[Callable[[Object], bool]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs,
    ) -> Union[
        Iterator[tuple[str, Union[Array, tuple[Array, Array]]]],
        Union[Array, tuple[Array, Array]],
    ]:
        """
        Repeatedly calls ``fun`` on all paths between the transmitters in the scene and every receiver coordinate in :python:`(X, Y)`, and accumulates the results over one array that has the same shape a ``X`` and ``Y``.

        Produces an iterator with one element for each transmitter location.

        :param X: The grid of x-coordinates.
        :param Y: The grid of y-coordinates.
        :param fun: The function to evaluate on each path.
        :param fun_args:
            Positional arguments passed to ``fun``.
        :param fun_kwargs:
            Keyword arguments passed to ``fun``.
        :param reduce_all: Whether to reduce the output by summing
            all accumulated results. This is especially useful
            if you only care about the total accumulated results.
        :param grad: If set, returns the gradient of ``fun`` with respect
            to the receiver's position. The output array(s) will then have
            an additional axis of size two.
        :param value_and_grad: If set, returns both the ``fun`` and its
            gradient. Takes precedence over setting :python:`grad=True`.
        :param path_cls: Method to be used to find the path coordinates.
        :param path_cls_kwargs:
            Keyword arguments passed to ``path_cls.from_tx_objects_rx``.
        :param receiver_cls: A point constructor called on every receiver,
            should inherit from :class:`Point<differt2d.geometry.Point>`.
        :param min_order:
            The minimum order of the path, i.e., the number of interactions.
        :param max_order:
            The maximum order of the path, i.e., the number of interactions.
        :param order: If provided, it is equivalent to setting
            ``min_order=order`` and ``max_order=order``.
        :param filter_objects: A callable indicating
            which objects should be used for path candidates.
        :param key: The random key to be used to find the paths.
            Depending on ``path_cls``, this can be mandatory.
        :param kwargs:
            Keyword arguments passed to
            :meth:`Path.is_valid<differt2d.geometry.Path.is_valid>`.
        :return:
            An iterator of transmitter name and the corresponding
            accumulated result, or the sum of accumulated results
            if :python:`reduce_all=True`.
        """
        if fun_kwargs is None:
            fun_kwargs = {}

        if path_cls_kwargs is None:
            path_cls_kwargs = {}

        scene = self.with_receivers(rx=Point(xy=jnp.array([0.0, 0.0])))

        path_candidates = scene.all_path_candidates(
            min_order=min_order,
            max_order=max_order,
            order=order,
            filter_objects=filter_objects,
        )

        pairs = list(scene.all_transmitter_receiver_pairs())

        if key is not None:
            keys = list(jax.random.split(key, len(path_candidates)))
        else:
            keys = [None] * len(path_candidates)

        def facc(transmitter: Point, rx_coords: Float[Array, "2"]) -> Float[Array, " "]:
            acc = jnp.array(0.0)
            for path_candidate, key in zip(path_candidates, keys):
                interacting_objects = scene.get_interacting_objects(path_candidate)
                path = path_cls.from_tx_objects_rx(
                    transmitter,
                    interacting_objects,
                    rx_coords,
                    key=key,
                    **path_cls_kwargs,
                )
                valid = path.is_valid(
                    scene.objects,
                    path_candidate,
                    interacting_objects,
                    **kwargs,
                )
                acc = acc + valid * fun(
                    transmitter,
                    receiver_cls(xy=rx_coords),
                    path,
                    interacting_objects,
                    *fun_args,
                    **fun_kwargs,
                )

            return acc

        if value_and_grad:
            f = jax.value_and_grad(facc, argnums=1)  # type: ignore
        elif grad:
            f = jax.grad(facc, argnums=1)
        else:
            f = facc

        vf = jax.vmap(
            jax.vmap(f, in_axes=(None, 0)),
            in_axes=(None, 0),
        )

        grid = jnp.dstack((X, Y))

        def results() -> Iterator[tuple[str, Union[Array, tuple[Array, Array]]]]:
            return (
                (tx_key, vf(transmitter, grid)) for (tx_key, transmitter), _ in pairs
            )

        if reduce_all:
            if value_and_grad:
                Z = jnp.array(0.0)
                dZ = jnp.array(0.0)
                for _, (p, dp) in results():
                    Z = Z + p
                    dZ = dZ + dp

                return Z, dZ
            else:
                Z = jnp.array(0.0)
                for _, p in results():
                    Z = Z + p

                return Z
        else:
            return results()
