from typing import List

import chex
import jax.numpy as jnp
from chex import Array

from .abc import Interactable, Plottable
from .logic import jit, jit_approx, ne_


@chex.dataclass
class Ray:
    points: Array

    def __post_init__(self):
        chex.assert_shape(self.points, (2, 2))

    @jit
    def origin(self) -> Array:
        return self.points[0]

    @jit
    def dest(self) -> Array:
        return self.points[1]

    @jit
    def t(self):
        return self.dest() - self.origin()


@chex.dataclass
class Point:
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


@chex.dataclass
class Wall(Ray, Interactable, Plottable):
    points: Array

    @jit
    def normal(self) -> Array:
        t = self.t()
        n = t.at[0].set(t[1])
        n = n.at[1].set(-t[0])
        return n / jnp.linalg.norm(n)

    @jit
    def parameters_count(self) -> int:
        return 1

    @chex.chexify
    @jit
    def parametric_to_cartesian(self, param_coords: Array) -> Array:
        chex.assert_shape(param_coords, (1,))
        pass

    @chex.chexify
    @jit
    def cartesian_to_parametric(self, carte_coords: Array) -> Array:
        chex.assert_shape(carte_coords, (2,))
        pass

    @chex.chexify
    @jit
    def intersects_cartesian(self, ray: Array) -> Array:
        chex.assert_shape(ray, (2, 2))
        pass

    @chex.chexify
    @jit
    def evaluate_cartesian(self, ray_path: Array) -> Array:
        chex.assert_shape(ray_path, (3, 2))
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


@chex.dataclass
class RIS(Wall):
    phi: Array = jnp.pi / 4

    @chex.chexify
    @jit
    def evaluate_cartesian(self, ray_path: Array) -> Array:
        chex.assert_shape(ray_path, (3, 2))
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
class Path:
    points: Array

    def __post_init__(self):
        chex.assert_shape(self.points, (None, 2))

    @jit
    def length(self) -> Array:
        vectors = jnp.diff(self.points, axis=0)
        lengths = jnp.linalg.norm(vectors, axis=1)

        return jnp.sum(lengths)

    def plot(self, ax, *args, color="orange", **kwargs):
        x, y = self.points.T
        return ax.plot(x, y, *args, color=color, **kwargs)

@chex.dataclass
class MinPath(Path):

    @classmethod
    def from_tx_objects_rx(cls, tx: Point, objects: List[Interactable], rx: Point) -> "MinPath":
        n = len(objects)
        n_unknowns = sum([obj.parameters_count() for obj in objects])

        @jit
        def loss(theta):
            cartesian_coords = jnp.empty((n + 2, 2))
            cartesian_coords = cartesian_coords.at[0].set(tx.point)
            cartesian_coords = cartesian_coords.at[-1].set(rx.point)
            i = 0
            for i, obj in enumerate(objects):
                param_coords = theta[i, i + obj.parameters_count()]
                cartesian_coords = cartesian_coords.at[i + 1].set(obj.parametric_to_cartesian(param_coords))
                i += obj.parameters_count()

            _loss = 0.0
            for i, obj in enumerate(objects):
                param_coords = theta[i, i + obj.parameters_count()]
                cartesian_coords = cartesian_coords.at[-1].set(obj.parametric_to_cartesian(param_coords))
                i += obj.parameters_count()


        theta = jnp.zeros(n_unknowns)

        points = jnp.row_stack([tx.point, rx.point])
        return MinPath(points=points)
