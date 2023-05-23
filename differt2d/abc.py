"""
Abtract classes to be implemented by the user.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jax import Array
    from matplotlib.axes import Axes
else:
    Array = Any
    Axes = Any


class Plottable(ABC):
    """
    Abstract class for any object that can be plotted using matplotlib.
    """

    @abstractmethod
    def plot(self, ax: Axes, *args: Any, **kwargs: Any) -> Any:
        """
        Plot this object on the given axes and returns the results.

        :param ax: The axes to plot on.
        :type ax: Axes
        :param args: Parameters to be passed to the plot function.
        :type args: Any
        :param kwargs: Keyword parameters to be passed to the plot function.
        :type kwargs: Any
        :return: The artist(s).
        :rtype: Union[:class:`matplotlib.artist.Artist`, List[matplotlib.artist.Artist]]
        """
        pass

    @abstractmethod
    def bounding_box(self) -> Array:
        """
        Returns the bounding box of this object.

        This is: [[min_x, min_y], [max_x, max_y]]

        :return: The min. and max. coordinates of this object.
        :rtype: [2, 2], Array
        """
        pass


class Interactable(ABC):
    """
    Abstract class for any object that a ray path can interact with.
    """

    @staticmethod
    @abstractmethod
    def parameters_count() -> int:
        """
        Returns how many parameters (s, t, ...) are needed to define an
        interaction point on this object.

        Typically, this equals to one for 2D surfaces.

        :return: The number of parameters.
        :rtype: int
        """
        pass

    @abstractmethod
    def parametric_to_cartesian(self, param_coords: Array) -> Array:
        """
        Converts parametric coordinates to cartesian coordinates.

        :param param_coords: Parametric coordinates.
        :type param_coords: [:meth:`parameters_count()`], Array
        :return: Cartesian coordinates.
        :rtype: [2], Array
        """
        pass

    @abstractmethod
    def cartesian_to_parametric(self, carte_coords: Array) -> Array:
        """
        Converts cartesian coordinates to parametric coordinates.

        :param carte_coords: Cartesian coordinates.
        :type carte_coords: [2], Array
        :return: Parametric coordinates.
        :rtype: [:meth:`parameters_count()`], Array
        """
        pass

    @abstractmethod
    def contains_parametric(self, param_coords: Array) -> Array:
        """
        Checks if the given coordinates are within the object.

        :param param_coords: Parametric coordinates.
        :type param_coords: [:meth:`parameters_count()`], Array
        :return: True if object contains these coordinates.
        :rtype: Array
        """
        pass

    @abstractmethod
    def intersects_cartesian(self, ray: Array) -> Array:
        """
        Ray intersection test on the current object.

        :param ray: Ray coordinates.
        :type ray: [2, 2], Array
        :return: True if it intersects.
        :rtype: Array
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

        :param ray_path: Ray path coordinates.
        :type ray_path: [3, 2], Array
        :return: Interaction score.
        :rtype: [], Array
        """
        pass
