"""
Geometrical objects to be placed in a :class:`differt2d.scene.Scene`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, List

import jax
import jax.numpy as jnp
from jax import jit

from .abc import Interactable, Plottable
from .logic import greater, greater_equal, less, less_equal, logical_and, logical_or
from .optimize import minimize_many_random_uniform

if TYPE_CHECKING:
    from dataclasses import dataclass

    from jax import Array
else:
    from chex import dataclass


@jit
def segments_intersect_including_extremities(P1, P2, P3, P4):
    A = P2 - P1
    B = P3 - P4
    C = P1 - P3
    a = B[1] * C[0] - B[0] * C[1]
    b = A[0] * C[1] - A[1] * C[0]
    d = A[1] * B[0] - A[0] * B[1]
    a = a / d
    b = b / d
    a = logical_and(greater_equal(a, 0.0), less_equal(a, 1.0))
    b = logical_and(greater_equal(b, 0.0), less_equal(b, 1.0))
    return logical_and(a, b)


@jit
def segments_intersect_excluding_extremities(P1, P2, P3, P4):
    A = P2 - P1
    B = P3 - P4
    C = P1 - P3
    a = B[1] * C[0] - B[0] * C[1]
    b = A[0] * C[1] - A[1] * C[0]
    d = A[1] * B[0] - A[0] * B[1]
    a = a / d
    b = b / d
    a = logical_and(greater(a, 0.0), less(a, 1.0))
    b = logical_and(greater(b, 0.0), less(b, 1.0))
    return logical_and(a, b)


@partial(jax.jit, inline=True)
def path_length(points: Array) -> Array:
    """
    Returns the length of the given path, with N points.

    :param points: An array of points, (N, 2).
    :return: The path length, ().
    """
    vectors = jnp.diff(points, axis=0)
    lengths = jnp.linalg.norm(vectors, axis=1)

    return jnp.sum(lengths)


@dataclass
class Ray(Plottable):
    """
    A ray object with origin and destination points.
    """

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
        return n / jnp.linalg.norm(n)

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
        return jnp.dot(self.t(), other) / jnp.dot(self.t(), self.t())

    @jit
    def contains_parametric(self, param_coords: Array) -> Array:
        ge = greater_equal(param_coords, 0.0)
        le = less_equal(param_coords, 1.0)
        return logical_and(ge, le)

    @jit
    def intersects_cartesian(
        self, ray: Array, patch: float = 0.05, include_extremities: bool = True
    ) -> Array:
        if include_extremities:
            return segments_intersect_including_extremities(
                self.origin() - patch * self.t(),
                self.dest() + patch * self.t(),
                ray[0, :],
                ray[1, :],
            )
        else:
            return segments_intersect_excluding_extremities(
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
        li = jnp.linalg.norm(i)  # Incident's length
        lr = jnp.linalg.norm(r)  # Reflected's length

        mode = "normalized"

        if mode == "normalized":
            i = i / li
            r = r / lr
            e = r - (i - 2 * jnp.dot(i, n) * n)
        elif mode == "multiplied":
            e = li * r - lr * (i - 2 * jnp.dot(i, n) * n)
        elif mode == "other":
            re = i - 2 * jnp.dot(i, n) * n
            e = jnp.cross(re, r)

        return jnp.dot(e, e)  # * 0.05


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

    def plot(self, ax, *args, **kwargs):
        kwargs.setdefault("color", "green")
        super().plot(ax, *args, **kwargs)


@dataclass
class Path(Plottable, ABC):
    """
    A path object with at least two points.
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
    @abstractmethod
    def from_tx_objects_rx(
        cls,
        tx: Any,
        objects: List[Any],
        rx: Any,
    ) -> "Path":
        """
        Returns a path from TX to RX, traversing each object in the list
        in the provided order.

        :param tx: The emitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :return: The resulting path.
        """
        pass

    @jit
    def length(self) -> Array:
        """
        Returns the length of this path.

        :return: The path length, ().
        """
        return path_length(self.points)

    @jit
    def on_objects(self, objects: List[Interactable]) -> Array:
        contains = jnp.array(True)
        for i, obj in enumerate(objects):
            param_coords = obj.cartesian_to_parametric(self.points[i + 1, :])
            # jax.debug.print("Contains: {obj}, {coords}", obj=obj, coords=param_coords)
            contains = logical_and(contains, obj.contains_parametric(param_coords))

        return contains

    @jit
    def intersects_with_objects(
        self, objects: List[Interactable], path_candidate: List[int]
    ) -> Array:
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
            [jnp.min(self.points, axis=1), jnp.max(self.points, axis=1)]
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
    @partial(jit, static_argnames=("cls", "steps"))
    def from_tx_objects_rx(
        cls,
        tx: Point,
        objects: List[Interactable],
        rx: Point,
        seed: int = 1234,
        steps: int = 400,
    ) -> "FermatPath":
        """
        Returns a path with minimal length.

        :param tx: The emitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :param seed: The random seed used to generate the start iteration.
        :param steps: The number of iterations performed by the minimizer.
        :return: The resulting path of the FPT method.
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

        key = jax.random.PRNGKey(seed)
        theta, _ = minimize_many_random_uniform(fun=loss_fun, n=n_unknowns, key=key)

        points = parametric_to_cartesian(objects, theta, n, tx.point, rx.point)

        return FermatPath(points=points, loss=path_loss(points))


@dataclass
class MinPath(Path):
    """
    A Path object that was obtain with the Min-Path-Tracing method.
    """

    @classmethod
    @partial(jit, static_argnames=("cls", "steps"))
    def from_tx_objects_rx(
        cls,
        tx: Point,
        objects: List[Interactable],
        rx: Point,
        seed: int = 1234,
        steps: int = 100,
    ) -> "MinPath":
        """
        Returns a path that minimizes the sum of interactions.

        :param tx: The emitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :param seed: The random seed used to generate the start iteration.
        :param steps: The number of iterations performed by the minimizer.
        :return: The resulting path of the MPT method.
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

        key = jax.random.PRNGKey(seed)
        theta, loss = minimize_many_random_uniform(fun=loss_fun, n=n_unknowns, key=key)

        points = parametric_to_cartesian(objects, theta, n, tx.point, rx.point)

        return MinPath(points=points, loss=loss)
