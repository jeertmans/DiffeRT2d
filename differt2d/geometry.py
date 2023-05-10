from abc import ABC, abstractmethod
from functools import cached_property

import chex
import jax
import jax.numpy as jnp


class Reflector(ABC):
    @abstractmethod
    def t_to_xy(self, t: chex.Array) -> chex.Array:
        """Returns the cartesian coordinates from local parametric coordinate t."""
        pass

    @abstractmethod
    def xy_to_t(self, xy: chex.Array) -> chex.Array:
        pass

    @abstractmethod
    def normal_at_t(self, t: chex.Array) -> chex.Array:
        """Returns the local normal vector."""
        pass


@chex.dataclass
class Wall(Reflector):
    path: chex.Array

    def __post_init__(self):
        chex.assert_shape(self.path, (2, 2))

    @cached_property
    def origin(self) -> chex.Array:
        return self.path[0]

    @cached_property
    def dest(self) -> chex.Array:
        return self.path[1]

    @cached_property
    def t(self):
        return self.dest - self.origin

    def __getitem__(self, item):
        return self.path[item]

    @chex.chexify
    @jax.jit
    def t_to_xy(self, t: chex.Array) -> chex.Array:
        chex.assert_shape(t, ())
        return self.origin + t * self.t

    @jax.jit
    def xy_to_t(self, xy):
        pass

    @chex.chexify
    @jax.jit
    def normal_at_t(self, t: chex.Array) -> chex.Array:
        chex.assert_shape(t, ())
        n = self.t.at[0].set(self.t[1])
        n = n.at[1].set(-self.t[0])
        return n / jnp.linalg.norm(n)

    @jax.jit
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
    @jax.jit
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
