"""
Utilities.
"""

from typing import Any, Callable, Iterable, Iterator, List, Optional, TypeVar, Union

import jax
import jax.numpy as jnp

Pytree = Union[list, tuple, dict]
T = TypeVar("T")
RecursiveIter = Union[T, Iterable["RecursiveIter"]]


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
    Unstacks the leaves of a Pytree.
    Reciprocal of :func:`stack_leaves`.

    :param pytrees: A Pytree.
    :return: A list of Pytrees,
        where each Pytree has the same structure as the input Pytree,
        but each leaf contains only one part of the original leaf.
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytrees)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves)]


def flatten(recursive_iter: RecursiveIter) -> Iterator[T]:
    """
    Flattens an iterator of possibly nested iterators
    into one generator.

    :param recursive_iter: The list to flatten.
    :return: The flattened generator.
    """
    if isinstance(recursive_iter, Iterable):
        for t in recursive_iter:
            yield from flatten(t)
    else:
        yield recursive_iter
