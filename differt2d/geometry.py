"""
Geometrical objects to be placed in a :class:`differt2d.scene.Scene`.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jit

from .abc import Interactable, Plottable
from .logic import greater_equal, less_equal, logical_and, logical_or, true_value
from .optimize import minimize_many_random_uniform

if TYPE_CHECKING:  # pragma: no cover
    from dataclasses import dataclass

    from jax import Array
else:
    from chex import dataclass


@partial(jax.jit, inline=True, static_argnames=["approx", "function"])
def segments_intersect(
    P1: Array, P2: Array, P3: Array, P4: Array, tol: float = 0.0, **kwargs: Any
) -> Array:
    r"""
    Checks whether two line segments intersect.

    The first segment is defined by points P1-P2, and the second by
    points P3-P4.

    If they exist, the intersection coordinates can expressed as either:

    .. math::

        P = \alpha \left(P_2 - P_1 \right) + P_1,

    or,

    .. math::

        P = \alpha \left(P_4 - P_3 \right) + P_3.

    For :math:`P` to exist, both :math:`\alpha` and :math:`\beta`
    must be in the range :math:`[0;1]`.

    :param P1:
        The coordinates of the first point of the first segment, (2,).
    :param P2:
        The coordinates of the second point of the first segment, (2,).
    :param P3:
        The coordinates of the first point of the second segment, (2,).
    :param P4:
        The coordinates of the second point of the second segment, (2,).
    :param tol:
        Relaxes the condition to :math:`[-\texttt{tol};1+\texttt{tol}]`.
    :param kwargs:
        Keyword arguments to be passed to logical functions.
    :return: Whether the two segments intersect, ().

    .. warning::

        Division by zero may occur if the two segments are colinear.

    :Examples:

    >>> from differt2d.geometry import segments_intersect
    >>> import jax.numpy as jnp
    >>> P1 = jnp.array([+0., +0.]); P2 = jnp.array([+1., +0.])
    >>> P3 = jnp.array([+.5, -1.]); P4 = jnp.array([+.5, +1.])
    >>> segments_intersect(P1, P2, P3, P4)
    Array(1., dtype=float32)
    >>> segments_intersect(P1, P2, P3, P4, approx=False)
    Array(True, dtype=bool)
    >>> segments_intersect(P1, P2, P3, P4, function="sigmoid")
    Array(1., dtype=float32)


    :References:

    Code inspired from "Graphics Gems III - 1st Edition", section IV.6.

    https://www.realtimerendering.com/resources/GraphicsGems/gemsiii/insectc.c
    http://www.graphicsgems.org/
    """
    A = P2 - P1
    B = P3 - P4
    C = P1 - P3
    a = B[1] * C[0] - B[0] * C[1]  # alpha numerator
    b = A[0] * C[1] - A[1] * C[0]  # beta numerator
    d = A[1] * B[0] - A[0] * B[1]  # both denominator
    kwargs_no_function = kwargs.copy()
    kwargs_no_function.pop("function", None)

    @jit
    def test(num, den):
        den_is_zero = den == 0.0
        den = jnp.where(den_is_zero, 1.0, den)
        t = jnp.where(den_is_zero, jnp.inf, num / den)
        return logical_and(
            greater_equal(t, -tol, **kwargs),
            less_equal(t, 1.0 + tol, **kwargs),
            **kwargs_no_function,
        )

    intersect = logical_and(test(a, d), test(b, d), **kwargs_no_function)

    return intersect


@partial(jax.jit, inline=True)
def path_length(points: Array) -> Array:
    """
    Returns the length of the given path, with N points.

    :param points: An array of points, (N, 2).
    :return: The path length, ().

    .. note::

        Currently, some epsilon value is added to each path segment to avoid
        division by zero in the gradient of this function. Hopefully, this
        should not be perceived by the user.

    :Examples:

    >>> from differt2d.geometry import path_length
    >>> import jax.numpy as jnp
    >>> points = jnp.array([[0., 0.], [1., 0.], [1., 1.], [0., 0.]])
    >>> path_length(points)  # 1 + 1 + sqrt(2)
    Array(3.4142137, dtype=float32)
    """
    vectors = jnp.diff(points, axis=0)
    vectors = vectors + jnp.finfo(points.dtype).eps
    lengths = jnp.linalg.norm(vectors, axis=1)

    return jnp.sum(lengths)


@partial(jax.jit, inline=True)
def normalize(vector: Array) -> Tuple[Array, Array]:
    """
    Normalizes a vector, and also returns its length.

    :param vector: A vector, (2,).
    :return: The normalized vector and its length, (2,) and ().

    :Examples:

    >>> from differt2d.geometry import normalize
    >>> import jax.numpy as jnp
    >>> vector = jnp.array([1., 1.])
    >>> normalize(vector)  # [1., 1.] / sqrt(2), sqrt(2)
    (Array([0.70710677, 0.70710677], dtype=float32),
     Array(1.4142135, dtype=float32))
    >>> zero = jnp.array([0., 0.])
    >>> normalize(zero)  # Special behavior at 0.
    (Array([0., 0.], dtype=float32), Array(1., dtype=float32))
    """
    length = jnp.linalg.norm(vector)
    length = jnp.where(length == 0.0, jnp.ones_like(length), length)

    return vector / length, length


@dataclass
class Ray(Plottable):
    """
    A ray object with origin and destination points.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Ray

        ax = plt.gca()
        ray = Ray(points=jnp.array([[0., 0.], [1., 1.]]))
        _ = ray.plot(ax)
        plt.show()
    """

    """Ray points (origin, dest)."""
    points: Array  # a b

    @partial(jit, inline=True)
    def origin(self) -> Array:
        """
        Returns the origin of this object.

        :return: The origin.
        """
        return self.points[0]

    @partial(jit, inline=True)
    def dest(self) -> Array:
        """
        Returns the destination of this object.

        :return: The destination.
        """
        return self.points[1]

    @partial(jit, inline=True)
    def t(self) -> Array:
        """
        Returns the direction vector of this object.

        :return: The direction vector.
        """
        return self.dest() - self.origin()

    def plot(self, ax, *args, **kwargs):
        kwargs.setdefault("color", "blue")
        x, y = self.points.T
        return ax.plot(x, y, *args, **kwargs)

    def bounding_box(self) -> Array:
        return jnp.row_stack(
            [jnp.min(self.points, axis=0), jnp.max(self.points, axis=0)]
        )


@dataclass
class Point(Plottable):
    """
    A point object defined by its coordinates.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Point

        ax = plt.gca()
        point = Point(point=jnp.array([0., 0.]))
        _ = point.plot(ax)
        plt.show()
    """

    point: Array
    """Cartesian coordinates."""

    def plot(self, ax, *args, **kwargs):
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("color", "red")
        return ax.plot(
            [self.point[0]],
            [self.point[1]],
            *args,
            **kwargs,
        )

    def bounding_box(self) -> Array:
        return jnp.row_stack([self.point, self.point])


@dataclass
class Wall(Ray, Interactable):
    """
    A wall object defined by its corners.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Wall

        ax = plt.gca()
        wall = Wall(points=jnp.array([[0., 0.], [1., 0.]]))
        _ = wall.plot(ax)
        plt.show()
    """

    @jit
    def normal(self) -> Array:
        """
        Returns the normal to the current wall,
        expressed in cartesian coordinates and
        normalized.

        :return: The normal, (2,)
        """
        t = self.t()
        n = t.at[0].set(t[1])
        n = n.at[1].set(-t[0])
        n, _ = normalize(n)
        return n

    @staticmethod
    @partial(jit, inline=True)
    def parameters_count() -> int:
        return 1

    @jit
    def parametric_to_cartesian(self, param_coords: Array) -> Array:
        return self.origin() + param_coords * self.t()

    @jit
    def cartesian_to_parametric(self, carte_coords: Array) -> Array:
        other = carte_coords - self.origin()
        squared_length = jnp.dot(self.t(), self.t())
        squared_length = jnp.where(squared_length == 0.0, 1.0, squared_length)
        return jnp.dot(self.t(), other) / squared_length

    @jit
    def contains_parametric(self, param_coords: Array) -> Array:
        ge = greater_equal(param_coords, 0.0)
        le = less_equal(param_coords, 1.0)
        return logical_and(ge, le)

    @jit
    def intersects_cartesian(
        self,
        ray: Array,
        patch: float = 0.0,
    ) -> Array:
        return segments_intersect(
            self.origin() - patch * self.t(),
            self.dest() + patch * self.t(),
            ray[0, :],
            ray[1, :],
        )

    @jit
    def evaluate_cartesian(self, ray_path: Array) -> Array:
        i = ray_path[1, :] - ray_path[0, :]  # Incident
        r = ray_path[2, :] - ray_path[1, :]  # Reflected
        n = self.normal()  # Normal
        i, _ = normalize(i)
        r, _ = normalize(r)
        e = r - (i - 2 * jnp.dot(i, n) * n)
        return jnp.dot(e, e)


@dataclass
class RIS(Wall):
    """
    A very basic Reflective Intelligent Surface (RIS) object.

    Here, we model a RIS such that the angle of reflection is constant
    with respect to its normal, regardless of the incident vector.
    """

    phi: Array = jnp.pi / 4
    """
    The constant angle of reflection.
    """

    @jit
    def evaluate_cartesian(self, ray_path: Array) -> Array:
        v2 = ray_path[2, :] - ray_path[1, :]
        n = self.normal()
        sinx = jnp.cross(n, v2)  # |v2| * sin(x)
        sina = jnp.linalg.norm(v2) * jnp.sin(self.phi)
        return (sinx - sina) ** 2  # + (cosx - cosa) ** 2

    def plot(self, ax, *args, **kwargs):  # pragma: no cover
        kwargs.setdefault("color", "green")
        super().plot(ax, *args, **kwargs)


@dataclass
class Path(Plottable):
    """
    A path object with at least two points.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Path

        ax = plt.gca()
        path = Path(points=jnp.array([[0., 0.], [.8, .2], [1., 1.]]))
        _ = path.plot(ax)
        plt.show()
    """

    points: Array
    """
    Array of cartesian coordinates.
    """

    loss: float = 0.0
    """
    The loss value for the given path.
    """

    @classmethod
    def from_tx_objects_rx(
        cls,
        tx: Point,
        objects: List[Interactable],
        rx: Point,
    ) -> "Path":
        """
        Returns a path from TX to RX, traversing each object in the list
        in the provided order.

        The present implementation will sample a point at :python:`t = 0.5`
        on each object.

        :param tx: The emitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :return: The resulting path.

        :Examples:

        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            import jax.numpy as jnp
            from differt2d.geometry import Path
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            path = Path.from_tx_objects_rx(scene.tx, scene.objects, scene.rx)
            _ = path.plot(ax)
            plt.show()
        """
        points = [obj.parametric_to_cartesian(0.5) for obj in objects]
        points = jnp.row_stack([tx.point, points, rx.point])
        return cls(points=points)

    @jit
    def length(self) -> Array:
        """
        Returns the length of this path.

        :return: The path length, ().
        """
        return path_length(self.points)

    @jit
    def on_objects(self, objects: List[Interactable]) -> Array:
        """
        Returns whether the path correctly passes on the objects.

        For each object i, it will check whether it contains the ith point
        in the path (start and end points are ignored).

        :param objects: The list of objects to check against.
        :return: Whether this path passes on the objects, ().
        """
        # TODO: allow to pass kwargs
        contains = true_value()
        for i, obj in enumerate(objects):
            param_coords = obj.cartesian_to_parametric(self.points[i + 1, :])
            # jax.debug.print("Contains: {obj}, {coords}", obj=obj, coords=param_coords)
            contains = logical_and(contains, obj.contains_parametric(param_coords))

        return contains

    @jit
    def intersects_with_objects(
        self, objects: List[Interactable], path_candidate: List[int]
    ) -> Array:
        """
        Returns whether the path intersects with any of the objects.

        The ``path_candidate`` argument is used to avoid checking for
        intersection on objects the path is allowed to pass on.

        :param objects: The list of objects in the scene.
        :param path_candidate: The object indices on which the path should pass.
        :return: Whether this path intersects any of the objects, ().
        """
        # TODO: allow to pass kwargs
        interacting_object_indices = [-1] + [i - 1 for i in path_candidate[1:-1]] + [-1]
        intersects = jnp.array(False)

        for i in range(self.points.shape[0] - 1):
            ray_path = self.points[i : i + 2, :]
            for obj_index, obj in enumerate(objects):
                ignore = jnp.logical_or(
                    obj_index == interacting_object_indices[i + 0],
                    obj_index == interacting_object_indices[i + 1],
                )
                intersects = jnp.where(
                    ignore,
                    intersects,
                    logical_or(intersects, obj.intersects_cartesian(ray_path)),
                )

        return intersects

    def plot(self, ax, *args, **kwargs):
        kwargs.setdefault("color", "orange")
        x, y = self.points.T
        return ax.plot(x, y, *args, **kwargs)

    def bounding_box(self) -> Array:
        return jnp.row_stack(
            [jnp.min(self.points, axis=0), jnp.max(self.points, axis=0)]
        )


@partial(jax.jit, static_argnames=("size",))
def parametric_to_cartesian_from_slice(obj, parametric_coords, start, size):
    parametric_coords = jax.lax.dynamic_slice(parametric_coords, (start,), (size,))
    return obj.parametric_to_cartesian(parametric_coords)


@partial(jit, static_argnames=("n",))
def parametric_to_cartesian(objects, parametric_coords, n, tx_coords, rx_coords):
    cartesian_coords = jnp.empty((n + 2, 2))
    cartesian_coords = cartesian_coords.at[0].set(tx_coords)
    cartesian_coords = cartesian_coords.at[-1].set(rx_coords)
    j = 0
    for i, obj in enumerate(objects):
        size = obj.parameters_count()
        cartesian_coords = cartesian_coords.at[i + 1].set(
            parametric_to_cartesian_from_slice(obj, parametric_coords, j, size)
        )
        j += size

    return cartesian_coords


@dataclass
class FermatPath(Path):
    """
    A Path object that was obtain with the Fermat's Principle Tracing method.
    """

    @classmethod
    @partial(jit, static_argnames=("cls", "steps", "optimizer"))
    def from_tx_objects_rx(
        cls,
        tx: Point,
        objects: List[Interactable],
        rx: Point,
        key: Optional[jax.random.PRNGKey] = None,
        seed: int = 1234,
        **kwargs: Any,
    ) -> "FermatPath":
        """
        Returns a path with minimal length.

        :param tx: The emitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :param key: The random key to generate the initial guess.
        :param seed: The random seed used to generate the start iteration,
            only used if :python:`key is None`.
        :param kwargs:
            Keyword arguments to be passed to
            :func:`minimize_many_random_uniform<differt2d.optimize.minimize_many_random_uniform>`.
        :return: The resulting path of the FPT method.

        :Examples:

        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            import jax.numpy as jnp
            from differt2d.geometry import FermatPath
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            path = FermatPath.from_tx_objects_rx(scene.tx, scene.objects, scene.rx)
            _ = path.plot(ax)
            plt.show()
        """
        n = len(objects)
        n_unknowns = sum([obj.parameters_count() for obj in objects])

        @jit
        def loss_fun(theta):
            cartesian_coords = parametric_to_cartesian(
                objects, theta, n, tx.point, rx.point
            )

            return path_length(cartesian_coords)

        @jit
        def path_loss(cartesian_coords):
            _loss = 0.0
            for i, obj in enumerate(objects):
                _loss += obj.evaluate_cartesian(cartesian_coords[i : i + 3, :])

            return _loss

        if key is None:
            key = jax.random.PRNGKey(seed)

        theta, _ = minimize_many_random_uniform(
            fun=loss_fun, n=n_unknowns, key=key, **kwargs
        )

        points = parametric_to_cartesian(objects, theta, n, tx.point, rx.point)

        return cls(points=points, loss=path_loss(points))


@dataclass
class MinPath(Path):
    """
    A Path object that was obtain with the Min-Path-Tracing method.
    """

    @classmethod
    @partial(jit, static_argnames=("cls", "steps", "optimizer"))
    def from_tx_objects_rx(
        cls,
        tx: Point,
        objects: List[Interactable],
        rx: Point,
        key: Optional[jax.random.PRNGKey] = None,
        seed: int = 1234,
        **kwargs: Any,
    ) -> "MinPath":
        """
        Returns a path that minimizes the sum of interactions.

        :param tx: The emitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :param key: The random key to generate the initial guess.
        :param seed: The random seed used to generate the start iteration,
            only used if :python:`key is None`.
        :param kwargs:
            Keyword arguments to be passed to
            :func:`minimize_many_random_uniform<differt2d.optimize.minimize_many_random_uniform>`.
        :return: The resulting path of the MPT method.

        :Examples:

        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            import jax.numpy as jnp
            from differt2d.geometry import MinPath
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            path = MinPath.from_tx_objects_rx(scene.tx, scene.objects, scene.rx)
            _ = path.plot(ax)
            plt.show()
        """
        n = len(objects)
        n_unknowns = sum(obj.parameters_count() for obj in objects)

        @jit
        def loss_fun(theta):
            cartesian_coords = parametric_to_cartesian(
                objects, theta, n, tx.point, rx.point
            )

            _loss = 0.0
            for i, obj in enumerate(objects):
                _loss += obj.evaluate_cartesian(cartesian_coords[i : i + 3, :])

            return _loss

        if key is None:
            key = jax.random.PRNGKey(seed)

        theta, loss = minimize_many_random_uniform(
            fun=loss_fun, n=n_unknowns, key=key, **kwargs
        )

        points = parametric_to_cartesian(objects, theta, n, tx.point, rx.point)

        return cls(points=points, loss=loss)
