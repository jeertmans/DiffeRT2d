"""Abtract classes to be implemented by the user."""

from __future__ import annotations

__all__ = [
    "Interactable",
    "Loc",
    "Object",
    "Plottable",
]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Protocol, Tuple

import jax.numpy as jnp

from .defaults import DEFAULT_PATCH

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

Loc = Literal["N", "E", "S", "W", "C", "NE", "NW", "SE", "SW"]
"""Literal type for all valid locations."""


class Plottable(Protocol):  # pragma: no cover
    """Protocol for any object that can be plotted using matplotlib."""

    @abstractmethod
    def plot(self, ax: Axes, *args: Any, **kwargs: Any) -> List[Artist]:
        """
        Plot this object on the given axes and returns the results.

        :param ax: The axes to plot on.
        :param args: Arguments to be passed to the plot function.
        :param kwargs: Keyword arguments to be passed to the plot function.
        :return: The artist(s).
        """
        pass

    @abstractmethod
    def bounding_box(self) -> Array:
        """
        Returns the bounding box of this object.

        This is: :python:`[[min_x, min_y], [max_x, max_y]]`.

        :return: The min. and max. coordinates of this object, (2, 2).
        """
        pass

    def grid(self, n: int = 50) -> Tuple[Array, Array]:
        """
        Returns a (mesh) grid that overlays the current object.

        :param n: The number of sample along one axis.
        :return: A tuple of (X, Y) coordinates.
        :rtype: ((n, n), (n, n)), typing.Tuple[jax.Array, jax.Array]
        """
        bounding_box = self.bounding_box()
        x = jnp.linspace(bounding_box[0, 0], bounding_box[1, 0], n)
        y = jnp.linspace(bounding_box[0, 1], bounding_box[1, 1], n)

        X, Y = jnp.meshgrid(x, y)
        return X, Y

    def center(self) -> Array:
        """
        Returns the center coordinates of this object.

        This is: :python:`[avg_x, avg_y]`.

        :return: The average coordinates of this object, (2,).
        """
        bounding_box = self.bounding_box()

        return 0.5 * (bounding_box[0, :] + bounding_box[1, :])

    def get_location(self, location: Loc) -> Array:
        """
        Returns the relative location within this object's extents.

        'N', 'E', 'S', 'W', 'C' stand, respectively for North, East, South, West, and
        center. You can also combine two letters to define one of the four corners.

        :param location: A literal denothing the location.
        :return: The location coordinates within this object's extents.
        """
        (xmin, ymin), (xmax, ymax) = self.bounding_box()
        xavg = 0.5 * (xmin + xmax)
        yavg = 0.5 * (ymin + ymax)

        try:
            x, y = dict(
                N=(xavg, ymax),
                E=(xmax, yavg),
                S=(xavg, ymin),
                W=(xmin, yavg),
                C=(xavg, yavg),
                NE=(xmax, ymax),
                NW=(xmin, ymax),
                SE=(xmax, ymin),
                SW=(xmin, ymin),
            )[location]

            return jnp.array([x, y])

        except KeyError as e:
            raise ValueError(f"Invalid location '{location}'") from e


class Interactable(Protocol):  # pragma: no cover
    """Protocol for any object that a ray path can interact with."""

    @staticmethod
    @abstractmethod
    def parameters_count() -> int:
        """
        Returns how many parameters (s, t, ...) are needed to define an interaction
        point on this object.

        Typically, this equals to one for 2D surfaces.

        :return: The number of parameters.
        """
        pass

    @abstractmethod
    def parametric_to_cartesian(self, param_coords: Array) -> Array:
        """
        Converts parametric coordinates to cartesian coordinates.

        :param param_coords: Parametric coordinates, (:meth:`parameters_count()`,).
        :return: Cartesian coordinates, (2,).
        """
        pass

    @abstractmethod
    def cartesian_to_parametric(self, carte_coords: Array) -> Array:
        """
        Converts cartesian coordinates to parametric coordinates.

        :param carte_coords: Cartesian coordinates, (2,).
        :return: Parametric coordinates, (:meth:`parameters_count()`,).
        """
        pass

    @abstractmethod
    def contains_parametric(
        self,
        param_coords: Array,
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Array:
        """
        Checks if the given coordinates are within the object.

        :param param_coords: Parametric coordinates, (:meth:`parameters_count()`,).
        :param approx: Whether approximation is enabled or not.
        :param kwargs: Keyword arguments to be passed to
            :func:`activation<differt2d.logic.activation>`.
        :return: True if object contains these coordinates, (),
        """
        pass

    @abstractmethod
    def intersects_cartesian(
        self,
        ray: Array,
        patch: float = DEFAULT_PATCH,
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Array:
        """
        Ray intersection test on the current object.

        :param ray: Ray coordinates, (2, 2).
        :param patch: The patch ratio, to virtually resize the object
            prior to intersection check. A ``patch`` value greater than ``1``
            indicates that the object is enlarged, and a value between ``0`` and
            ``1`` indicates that the object is compressed. Patching the object
            size can be useful when combined with :python:`approx=True`, because
            smoothing objects can virtually reduce this object's size, so using
            a ``patch`` value greater than ``1`` can compensate this effect.
        :param approx: Whether approximation is enabled or not.
        :param kwargs: Keyword arguments to be passed to
            :func:`activation<differt2d.logic.activation>`.
        :return: True if it intersects, ().
        """
        pass

    @abstractmethod
    def evaluate_cartesian(self, ray_path: Array) -> Array:
        """
        Evaluates the given interaction triplet, such that:

        * incident vector is defined as :code:`v_in = b - a`;
        * bouncing vector is defined as :code:`v_out = c - b`;

        with :code:`a, b, c = ray_path` and :code:`b` lies on the current object.

        A return value of 0 indicates that the interaction is successful.

        The returned value cannot be negative.

        :param ray_path: Ray path coordinates, (3, 2).
        :return: Interaction score, ().
        """
        pass


class Object(Plottable, Interactable, Protocol):
    """
    Protocol for any object implementing both :class:`Plottable` and
    :class:`Interactable`.

    This type is actually needed to please Python type checkers, since using
    :python:`typing.Union[Plottable, Interactable]` is understood as implementing one of
    either protocols, not both.
    """

    pass
