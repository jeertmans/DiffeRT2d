"""Utilities."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from .abc import Interactable
    from .geometry import Path, Point

Pytree = Union[list, tuple, dict]
T = TypeVar("T")

DEFAULT_R_COEF: float = 0.5
"""Default value for real reflection coefficient."""

DEFAULT_HEIGHT: float = 0.1
"""Default TX antenna height, used to avoid division by zero when computing
:func:`received_power`."""

P0 = 1 / (DEFAULT_HEIGHT * DEFAULT_HEIGHT)
"""Default received power at zero distance from emitter, see :func:`received_power`."""


def stack_leaves(
    pytrees: Pytree, axis: int = 0, is_leaf: Optional[Callable[..., Any]] = None
) -> Pytree:
    """
    Stacks the leaves of one or more Pytrees along a new axis.

    Solution inspired from:
    https://github.com/google/jax/discussions/16882#discussioncomment-6638501.

    :param pytress: One or more Pytrees.
    :param axis: Axis along which leaves are stacked.
    :param is_leaf: See eponym parameter from :func:`jax.tree_util.tree_map`.
    :return: A new Pytree with leaves stacked along the new axis.
    """
    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=axis), *pytrees, is_leaf=is_leaf
    )


def unstack_leaves(pytrees) -> List[Pytree]:
    """
    Unstacks the leaves of a Pytree. Reciprocal of :func:`stack_leaves`.

    :param pytrees: A Pytree.
    :return: A list of Pytrees, where each Pytree has the same structure as the input
        Pytree, but each leaf contains only one part of the original leaf.
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytrees)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves)]


@jax.jit
def received_power(
    emitter: Point,
    receiver: Point,
    path: Path,
    interacting_objects: Sequence[Interactable],
    r_coef: float = DEFAULT_R_COEF,
    height: float = DEFAULT_HEIGHT,
) -> Array:
    """
    Returns the received power for a given path between some emitter and some receiver.

    Here, the power decreases with the square of the path length, and each interaction
    introduces some reflection coefficient.

    :param emitter: The emitting node, ignored.
    :param receiver: The receiving node, ignored.
    :param path: The ray path.
    :param interacting_objects: The sequence of interacting objects, ignored.
    :param r_coef: The reflection coefficient, with :python:`0 <= r_coef <= 1`.
    :param height: The TX antenna height to avoid division by zero when emitter and
        receiver are located at the same coordinates.
    :return: The received power.
    """
    r = path.length()
    n = path.points.shape[0] - 2  # Number of interactions
    return (r_coef**n) / (height * height + r * r)
