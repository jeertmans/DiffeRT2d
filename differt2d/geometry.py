from abc import ABC, abstractmethod
from functools import partial

import chex
import jax.numpy as jnp
from chex import Array
from jax.tree_util import register_pytree_node_class

from .logic import jit, jit_approx, ne_


class Reflector(ABC):
    @abstractmethod
    def t_to_xy(self, t: Array) -> Array:
        """Returns the cartesian coordinates from local parametric coordinate t."""
        pass

    @abstractmethod
    def xy_to_t(self, xy: Array) -> Array:
        pass

    @abstractmethod
    def normal_at_t(self, t: Array) -> Array:
        """Returns the local normal vector."""
        pass


class Interaction(ABC):
    @abstractmethod
    def evaluate_cartesian(self, a: Array, b: Array, c: Array) -> float:
        """
        Evaluates the given interaction triplet, such that:
        - incident vector is defined as v_in = b - a;
        - bouncing vector is defined as v_out = c - b;
        where b lies on the current object.

        a, b, and c are 2d-points in the cartesian coordinate space.

        A return value of 0 indicates that the interaction is successful.

        The returned value cannot be negative.
        """
        pass


@chex.dataclass
class Ray:
    path: Array

    def __post_init__(self):
        chex.assert_shape(self.path, (2, 2))

    @jit
    def origin(self) -> Array:
        return self.path[0]

    @jit
    def dest(self) -> Array:
        return self.path[1]

    @jit
    def t(self):
        return self.dest() - self.origin()

    def __getitem__(self, item):
        return self.path[item]


@chex.dataclass
class Wall(Ray, Reflector):
    @chex.chexify
    @jit
    def t_to_xy(self, t: Array) -> Array:
        chex.assert_shape(t, ())
        return self.origin() + t * self.t()

    @jit
    def xy_to_t(self, xy):
        pass

    @chex.chexify
    @jit
    def normal_at_t(self, t: Array) -> Array:
        chex.assert_shape(t, ())
        t = self.t()
        n = t.at[0].set(t[1])
        n = n.at[1].set(-t[0])
        return n / jnp.linalg.norm(n)

    @jit
    def interaction(self, v1, v2, n, **kwargs):
        i = jnp.linalg.norm(v1) * v2 - (
            v1 - 2 * (jnp.dot(v1, n) * n)
        ) * jnp.linalg.norm(v2)

        return jnp.dot(i, i)

    def plot(self, ax, *args, **kwargs):
        x, y = self.path.T
        return ax.plot(x, y, *args, **kwargs)


@chex.dataclass
class RIS(Wall):
    @jit
    def interaction(self, v1, v2, n, phi=jnp.pi / 4, **kwargs):
        sinx = jnp.cross(n, v2)  # |v2| * sin(x)
        sina = jnp.linalg.norm(v2) * jnp.sin(phi)
        # cosx = jnp.dot(n, v2)  # |v2| * cos(x)
        # cosa = jnp.linalg.norm(v2) * jnp.cos(alpha_ris)
        return (sinx - sina) ** 2  # + (cosx - cosa) ** 2

    def plot(self, ax, *args, **kwargs):
        if "color" in kwargs:
            del kwargs["color"]
        super().plot(ax, *args, color="green", **kwargs)
