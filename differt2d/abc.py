"""
Abtract classes to be implemented by the user.
"""

from __future__ import annotations

__all__ = [
    "Interactable",
    "Plottable",
]

from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Literal, Protocol, Union

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

LOC = Literal["N", "E", "S", "W", "C", "NE", "NW", "SE", "SW"]


class LocEnum(str, Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"
    C = "C"
    NE = "NE"
    NW = "NW"
    SE = "SE"
    SW = "SW"


class Plottable(Protocol):  # pragma: no cover
    """
    Protocol for any object that can be plotted using matplotlib.
    """

    @abstractmethod
    def plot(self, ax: Axes, *args: Any, **kwargs: Any) -> Union[Artist, List[Artist]]:
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

    def center(self) -> Array:
        """
        Returns the center coordinates of this object.

        This is: :python:`[avg_x, avg_y]`.

        :return: The average coordinates of this object, (2,).
        """
        bounding_box = self.bounding_box()

        return 0.5 * (bounding_box[0, :] + bounding_box[1, :])

    def get_location(self, location: LOC) -> Array:
        """
        Returns the relative location within this object's extents.

        'N', 'E', 'S', 'W', 'C' stand, respectively for North, East,
        South, West, and center. You can also combine two letters
        to define one of the four corners.

        :param location: A literal denothing the location.
        :return: The location coordinates within this
            object's extents.
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
    """
    Protocol for any object that a ray path can interact with.
    """

    @staticmethod
    @abstractmethod
    def parameters_count() -> int:
        """
        Returns how many parameters (s, t, ...) are needed to define an
        interaction point on this object.

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
    def contains_parametric(self, param_coords: Array) -> Array:
        """
        Checks if the given coordinates are within the object.

        :param param_coords: Parametric coordinates, (:meth:`parameters_count()`,).
        :return: True if object contains these coordinates, (),
        """
        pass

    @abstractmethod
    def intersects_cartesian(self, ray: Array) -> Array:
        """
        Ray intersection test on the current object.

        :param ray: Ray coordinates, (2, 2).
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
