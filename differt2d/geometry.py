from functools import partial
from typing import TYPE_CHECKING, Any, List

import chex
import jax
import jax.numpy as jnp
import optax

from .abc import Interactable, Plottable
from .logic import jit

if TYPE_CHECKING:
    from jax import Array
else:
    Array = Any


@jit
def segments_intersect(P1, P2, P3, P4):
    A = P2 - P1
    B = P3 - P4
    C = P1 - P3
    a = B[1] * C[0] - B[0] * C[1]
    b = A[0] * C[1] - A[1] * C[0]
    d = A[1] * B[0] - A[0] * B[1]
    a = a / d
    b = b / d
    a = jnp.logical_and(jnp.greater(a, 0.0), jnp.less(a, 1.0))
    b = jnp.logical_and(jnp.greater(b, 0.0), jnp.less(b, 1.0))
    return jnp.logical_and(a, b)


@chex.dataclass
class Ray:
    points: Array

    def __post_init__(self):
        chex.assert_shape(self.points, (2, 2))

    @partial(jit, inline=True)
    def origin(self) -> Array:
        return self.points[0]

    @partial(jit, inline=True)
    def dest(self) -> Array:
        return self.points[1]

    @partial(jit, inline=True)
    def t(self):
        return self.dest() - self.origin()


@chex.dataclass
class Point(Plottable):
    point: Array

    def __post_init__(self):
        chex.assert_shape(self.point, (2,))

    def plot(self, ax, *args, marker="o", color="red", **kwargs):
        return ax.plot(
            [self.point[0]],
            [self.point[1]],
            *args,
            marker=marker,
            color=color,
            **kwargs,
        )

    def bounding_box(self) -> Array:
        return jnp.row_stack([self.point, self.point])


@chex.dataclass
class Wall(Ray, Interactable, Plottable):
    @jit
    def normal(self) -> Array:
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
        ge = jnp.greater_equal(param_coords, 0.0)
        le = jnp.less_equal(param_coords, 1.0)
        return jnp.logical_and(ge, le)

    @jit
    def intersects_cartesian(self, ray: Array) -> Array:
        return segments_intersect(self.origin(), self.dest(), ray[0, :], ray[1, :])

    @jit
    def evaluate_cartesian(self, ray_path: Array) -> Array:
        v1 = ray_path[1, :] - ray_path[0, :]
        v2 = ray_path[2, :] - ray_path[1, :]
        n = self.normal()
        i = jnp.linalg.norm(v1) * v2 - (
            v1 - 2 * (jnp.dot(v1, n) * n)
        ) * jnp.linalg.norm(v2)

        return jnp.dot(i, i)

    def plot(self, ax, *args, color="blue", **kwargs):
        x, y = self.points.T
        return ax.plot(x, y, *args, color=color, **kwargs)

    def bounding_box(self) -> Array:
        return jnp.row_stack([jnp.min(self.points), jnp.max(self.points)])


@chex.dataclass
class RIS(Wall):
    phi: Array = jnp.pi / 4

    @jit
    def evaluate_cartesian(self, ray_path: Array) -> Array:
        v2 = ray_path[2, :] - ray_path[1, :]
        n = self.normal()
        sinx = jnp.cross(n, v2)  # |v2| * sin(x)
        sina = jnp.linalg.norm(v2) * jnp.sin(self.phi)
        return (sinx - sina) ** 2  # + (cosx - cosa) ** 2

    def plot(self, ax, *args, **kwargs):
        if "color" in kwargs:
            del kwargs["color"]
        super().plot(ax, *args, color="green", **kwargs)


@chex.dataclass
class Path(Plottable):
    points: Array

    def __post_init__(self):
        chex.assert_shape(self.points, (None, 2))

    @jit
    def length(self) -> Array:
        vectors = jnp.diff(self.points, axis=0)
        lengths = jnp.linalg.norm(vectors, axis=1)

        return jnp.sum(lengths)

    @jit
    def on_objects(self, objects: List[Interactable]) -> Array:
        contains = jnp.array(True)
        for i, obj in enumerate(objects):
            param_coords = obj.cartesian_to_parametric(self.points[i + 1, :])
            contains = jnp.logical_and(contains, obj.contains_parametric(param_coords))

        return contains

    @jit
    def intersects_with_objects(self, objects: List[Interactable]) -> Array:
        intersects = jnp.array(False)
        for obj in objects:
            for i in range(self.points.shape[0] - 1):
                ray_path = self.points[i : i + 2, :]
                intersects = jnp.logical_or(
                    intersects, obj.intersects_cartesian(ray_path)
                )

        return intersects

    def plot(self, ax, *args, color="orange", **kwargs):
        x, y = self.points.T
        return ax.plot(x, y, *args, color=color, **kwargs)

    def bounding_box(self) -> Array:
        return jnp.row_stack([jnp.min(self.points), jnp.max(self.points)])


@partial(jax.jit, static_argnums=(3,))
def parametric_to_cartesian_from_slice(obj, parametric_coords, start, size):
    parametric_coords = jax.lax.dynamic_slice(parametric_coords, (start,), (size,))
    return obj.parametric_to_cartesian(parametric_coords)


@chex.dataclass
class MinPath(Path):
    loss: float

    @classmethod
    @partial(jit, static_argnums=(0,))
    def from_tx_objects_rx(
        cls,
        tx: Point,
        objects: List[Interactable],
        rx: Point,
        seed: int = 1234,
        steps: int = 200,
    ) -> "MinPath":
        """
        Returns a path that minimizes the sum of interactions.
        """
        n = len(objects)
        n_unknowns = sum([obj.parameters_count() for obj in objects])

        @jit
        def parametric_to_cartesian(parametric_coords):
            cartesian_coords = jnp.empty((n + 2, 2))
            cartesian_coords = cartesian_coords.at[0].set(tx.point)
            cartesian_coords = cartesian_coords.at[-1].set(rx.point)
            j = 0
            for i, obj in enumerate(objects):
                size = obj.parameters_count()
                cartesian_coords = cartesian_coords.at[i + 1].set(
                    parametric_to_cartesian_from_slice(obj, parametric_coords, j, size)
                )
                j += size

            return cartesian_coords

        @jit
        def loss(theta):
            cartesian_coords = parametric_to_cartesian(theta)

            _loss = 0.0
            for i, obj in enumerate(objects):
                _loss += obj.evaluate_cartesian(cartesian_coords[i : i + 3, :])

            return _loss

        key = jax.random.PRNGKey(seed)
        optimizer = optax.adam(learning_rate=1)
        theta = jax.random.uniform(key, shape=(n_unknowns,))

        f_and_df = jax.value_and_grad(loss)
        opt_state = optimizer.init(theta)

        def f(carry, x):
            theta, opt_state = carry
            loss, grads = f_and_df(theta)
            updates, opt_state = optimizer.update(grads, opt_state)
            theta = theta + updates
            carry = (theta, opt_state)
            return carry, loss

        (theta, _), losses = jax.lax.scan(
            f, init=(theta, opt_state), xs=None, length=steps
        )

        points = parametric_to_cartesian(theta)

        return MinPath(points=points, loss=losses[-1])
