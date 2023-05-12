"""
Abtract classes to be implemented by the user.
"""
from abc import ABC, abstractmethod

import chex
from chex import Array


class Interactable(ABC):
    """
    Abstract class for any object that a ray path can interact with.
    """

    @abstractmethod
    def parameters_count(self) -> int:
        """
        Returns how many parameters (s, t, ...) are needed to define an
        interaction point on this object.

        Returns
        -------

        count: int
            The number of parameters.
        """
        pass

    @abstractmethod
    def parametric_to_cartesian(self, param_coords: Array) -> Array:
        """
        Converts parametric coordinates to cartesian coordinates.

        Parameters
        ----------

        param_coords: [self.parameters_count()], Array
            Parametric coordinates.

        Returns
        -------

        carte_coords: [2], Array
            Cartesian coordinates.
        """
        pass

    @abstractmethod
    def cartesian_to_parametric(self, carte_coords: Array) -> Array:
        """
        Converts cartesian coordinates to parametric coordinates.

        Parameters
        ----------

        cartes_coords: [2], Array
            Cartesian coordinates.

        Returns
        -------

        param_coords: [self.parameters_count()], Array
            Parametric coordinates.
        """
        pass

    @abstractmethod
    def intersects_cartesian(self, ray: Array) -> Array:
        """
        Ray intersection test on the current object.

        Parameters
        ----------

        ray: [2, 2], Array
            Ray coordinates.

        Returns
        -------

        intersects: Array
            True if it intersects.
        """
        pass

    @abstractmethod
    def evaluate_cartesian(self, ray_path: Array) -> Array:
        """
        Evaluates the given interaction triplet, such that:
        - incident vector is defined as v_in = b - a;
        - bouncing vector is defined as v_out = c - b;
        where b lies on the current object.

        a, b, and c are 2d-points in the cartesian coordinate space.

        A return value of 0 indicates that the interaction is successful.

        The returned value cannot be negative.

        Parameters
        ----------

        ray_path: [3, 2], Array
            Ray path coordinates.

        Returns
        -------

        score: [], Array
            Interaction score.
        """
        pass


@chex.dataclass
class Wall(Interactable):
    points: Array

    def parameters_count(self) -> int:
        return 1

    def parametric_to_cartesian(self, param_coords: Array) -> Array:
        pass

    def cartesian_to_parametric(self, carte_coords: Array) -> Array:
        pass

    def intersects_cartesian(self, ray: Array) -> Array:
        pass

    def evaluate_cartesian(self, ray_path: Array) -> Array:
        pass
