"""
Abtract classes to be implemented by the user.
"""

from __future__ import annotations

__all__ = [
    "Interactable",
    "Parametric",
    "Plottable",
]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Protocol, Union

if TYPE_CHECKING:
    from jax import Array
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes


class Plottable(Protocol):
    """
    Protocol for any object that can be plotted using matplotlib.
    """

    @abstractmethod
    def plot(self, ax: Axes, *args: Any, **kwargs: Any) -> Union[Artist, List[Artist]]:
        """
        Plot this object on the given axes and returns the results.

        :param ax: The axes to plot on.
        :param args: Parameters to be passed to the plot function.
        :param kwargs: Keyword parameters to be passed to the plot function.
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


class Parametric(Protocol):
    """
    Protocol for any object that can be expressed using parametric coordinates.
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


class Interactable(Protocol):
    """
    Protocol for any object that a ray path can interact with.
    """

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
