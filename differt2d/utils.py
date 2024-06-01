"""Utilities."""

from collections.abc import Sequence

import jax
from jaxtyping import Array, Float

from .abc import Interactable
from .defaults import DEFAULT_HEIGHT, DEFAULT_R_COEF
from .geometry import Path, Point

P0: float = 100.0
"""Received power at zero distance from transmitter when using default
parameter values, see :func:`received_power`."""


@jax.jit
def received_power(
    transmitter: Point,
    receiver: Point,
    path: Path,
    interacting_objects: Sequence[Interactable],
    r_coef: float = DEFAULT_R_COEF,
    height: float = DEFAULT_HEIGHT,
) -> Float[Array, " "]:
    """
    Returns the received power for a given path between some transmitter and some receiver.

    Here, the power decreases with the square of the path length, and
    each interaction introduces some reflection coefficient.

    :param transmitter: The transmitting node, ignored.
    :param receiver: The receiving node, ignored.
    :param path: The ray path.
    :param interacting_objects: The sequence of interacting objects,
        ignored.
    :param r_coef: The reflection coefficient, with :python:`0 <= r_coef
        <= 1`.
    :param height: The TX antenna height to avoid division by zero when
        transmitter and receiver are located at the same coordinates.
    :return: The received power.
    """
    r = path.length()
    n = path.xys.shape[0] - 2  # Number of interactions
    return (r_coef**n) / (height * height + r * r)
