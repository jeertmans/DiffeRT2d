"""Geometrical objects to be placed in a :class:`differt2d.scene.Scene`."""

from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from functools import partial
from typing import Any, Callable, Optional, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, jaxtyped
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from ._typing import ScalarFloat, ScalarInt
from .abc import Interactable, Object, Plottable
from .defaults import DEFAULT_PATCH
from .logic import (
    Truthy,
    false_value,
    greater_equal,
    less,
    less_equal,
    logical_all,
    logical_and,
    logical_not,
    logical_or,
    true_value,
)
from .optimize import minimize_many_random_uniform

T = TypeVar("T")
P = TypeVar("P", bound=PyTree)


@eqx.filter_jit
def stack_leaves(
    pytrees: Iterable[P],
    axis: int = 0,
    is_leaf: Optional[Callable[..., Any]] = None,
) -> P:
    """
    Stacks the leaves of one or more Pytrees along a new axis.

    Solution inspired from:
    https://github.com/google/jax/discussions/16882#discussioncomment-6638501.

    :param pytrees: One or more Pytrees.
    :param axis: Axis along which leaves are stacked.
    :param is_leaf: See eponym parameter from :func:`jax.tree_util.tree_map`.
    :return: A new Pytree with leaves stacked along the new axis.
    """
    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=axis), *pytrees, is_leaf=is_leaf
    )


@eqx.filter_jit
def unstack_leaves(pytrees) -> list[PyTree]:
    """
    Unstacks the leaves of a Pytree. Reciprocal of :func:`stack_leaves`.

    :param pytrees: A Pytree.
    :return: A list of Pytrees, where each Pytree has the same structure
        as the input Pytree, but each leaf contains only one part of the
        original leaf.
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytrees)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves)]


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def segments_intersect(
    P1: Float[Array, "2"],
    P2: Float[Array, "2"],
    P3: Float[Array, "2"],
    P4: Float[Array, "2"],
    tol: ScalarFloat = 0.005,
    approx: Optional[bool] = None,
    **kwargs: Any,
) -> Truthy:
    r"""
    Checks whether two line segments intersect.

    The first segment is defined by points P1-P2, and the second by
    points P3-P4.

    If they exist, the intersection coordinates can expressed as either:

    .. math::

        P = \alpha \left(P_2 - P_1 \right) + P_1,

    or,

    .. math::

        P = \alpha \left(P_4 - P_3 \right) + P_3.

    For :math:`P` to exist, both :math:`\alpha` and :math:`\beta`
    must be in the range :math:`[0;1]`.

    .. warning::

        Division by zero may occur if the two segments are colinear.

    :References:

      `"Graphics Gems III - 1st Edition", section IV.6 <http://www.graphicsgems.org/>`_. `Link to code <https://www.realtimerendering.com/resources/GraphicsGems/gemsiii/insectc.c>`_.

    :param P1:
        The coordinates of the first point of the first segment.
    :param P2:
        The coordinates of the second point of the first segment.
    :param P3:
        The coordinates of the first point of the second segment.
    :param P4:
        The coordinates of the second point of the second segment.
    :param tol:
        Relaxes the condition to :math:`[-\texttt{tol};1+\texttt{tol}]`.
    :param approx: Whether approximation is enabled or not.
    :param kwargs:
        Keyword arguments passed to :func:`activation<differt2d.logic.activation>`.
    :return: Whether the two segments intersect.

    :Examples:

    >>> from differt2d.geometry import segments_intersect
    >>> from differt2d.logic import sigmoid
    >>> import jax.numpy as jnp
    >>> P1 = jnp.array([+0.0, +0.0])
    >>> P2 = jnp.array([+1.0, +0.0])
    >>> P3 = jnp.array([+0.5, -1.0])
    >>> P4 = jnp.array([+0.5, +1.0])
    >>> segments_intersect(P1, P2, P3, P4, approx=True)
    Array(1., dtype=float32)
    >>> segments_intersect(P1, P2, P3, P4, approx=False)
    Array(True, dtype=bool)
    >>> segments_intersect(P1, P2, P3, P4, approx=True, function=sigmoid)
    Array(1., dtype=float32)
    """
    tol = jnp.asarray(tol)
    A = P2 - P1
    B = P3 - P4
    C = P1 - P3
    a = B[1] * C[0] - B[0] * C[1]  # alpha numerator
    b = A[0] * C[1] - A[1] * C[0]  # beta numerator
    d = A[1] * B[0] - A[0] * B[1]  # both denominator

    @partial(jax.jit, inline=True)
    def test(num, den):
        den_is_zero = den == 0.0
        den = jnp.where(den_is_zero, 1.0, den)
        t = jnp.where(den_is_zero, jnp.inf, num / den)
        return logical_and(
            greater_equal(t, -tol, approx=approx, **kwargs),
            less_equal(t, 1.0 + tol, approx=approx, **kwargs),
            approx=approx,
        )

    intersect = logical_and(test(a, d), test(b, d), approx=approx)

    return intersect


@partial(jax.jit, inline=True)
@jaxtyped(typechecker=typechecker)
def path_length(points: Float[Array, "N 2"]) -> Float[Array, " "]:
    """
    Returns the length of the given path, with N points.

    .. note::

        Currently, some epsilon value is added to each path segment to avoid
        division by zero in the gradient of this function. Hopefully, this
        should not be perceived by the user.

    :param points: An array of points.
    :return: The path length.

    :Examples:

    >>> from differt2d.geometry import path_length
    >>> import jax.numpy as jnp
    >>> points = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    >>> path_length(points)  # 1 + 1 + sqrt(2)
    Array(3.4142137, dtype=float32)
    """
    vectors = jnp.diff(points, axis=0)
    vectors = vectors + jnp.finfo(points.dtype).eps
    lengths = jnp.linalg.norm(vectors, axis=1)

    return jnp.sum(lengths)


@partial(jax.jit, inline=True)
@jaxtyped(typechecker=typechecker)
def normalize(vector: Float[Array, "2"]) -> tuple[Float[Array, "2"], Float[Array, " "]]:
    """
    Normalizes a vector, and also returns its length.

    :param vector: A vector.
    :return: The normalized vector and its length.

    :Examples:

    >>> from differt2d.geometry import normalize
    >>> import jax.numpy as jnp
    >>> vector = jnp.array([1.0, 1.0])
    >>> normalize(vector)  # [1., 1.] / sqrt(2), sqrt(2)
    (Array([0.70710677, 0.70710677], dtype=float32),
     Array(1.4142135, dtype=float32))
    >>> zero = jnp.array([0.0, 0.0])
    >>> normalize(zero)  # Special behavior at 0.
    (Array([0., 0.], dtype=float32), Array(1., dtype=float32))
    """
    length = jnp.linalg.norm(vector)
    length: Float[Array, " "] = jnp.where(length == 0.0, jnp.ones_like(length), length)

    return vector / length, length


@partial(jax.jit, inline=True)
@jaxtyped(typechecker=typechecker)
def closest_point(
    points: Float[Array, "N 2"], target: Float[Array, "2"]
) -> tuple[Int[Array, " "], Float[Array, " "]]:
    """
    Returns the index of the closest point to some target, and the actual distance.

    :param points: An array of 2D points.
    :param target: A target point.
    :return: The index of the closest point and the distance to the target.

    :Examples:

    >>> from differt2d.geometry import closest_point
    >>> import jax.numpy as jnp
    >>> target = jnp.array([0.6, 0.3])
    >>> points = jnp.array(
    ...     [
    ...         [0.0, 0.0],
    ...         [1.0, 0.0],  # This is the closest point
    ...         [1.0, 1.0],
    ...         [0.0, 1.0],
    ...     ]
    ... )
    >>> closest_point(points, target)
    (Array(1, dtype=int32), Array(0.49999997, dtype=float32))
    >>> points[closest_point(points, target)[0]]
    Array([1., 0.], dtype=float32)
    """
    distances = jnp.linalg.norm(points - target.reshape(-1, 2), axis=1)
    i_min = jnp.argmin(distances)

    return i_min, distances[i_min]


@jaxtyped(typechecker=typechecker)
class Point(Plottable, eqx.Module):
    """
    A point object defined by its coordinates.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Point

        ax = plt.gca()
        p1 = Point(xy=jnp.array([0., 0.]))
        _ = p1.plot(ax)
        p2 = Point(xy=jnp.array([1., 1.]))
        _ = p2.plot(ax, color="b", annotate="$p_2$")
        plt.show()  # doctest: +SKIP

    """

    xy: Float[Array, "2"] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.zeros(2)
    )
    """Cartesian coordinates."""

    def plot(
        self,
        ax: Axes,
        *args: Any,
        annotate: Optional[str] = None,
        annotate_offset: tuple[float, float] = (0.0, 0.0),
        annotate_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> MutableSequence[Artist]:
        """
        :param annotate: Text to put next the the point.
        :param annotate_offset:
        :param annotate_kwargs:
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.annotate`.
        """  # noqa: D205
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("color", "red")

        if annotate_kwargs is None:
            annotate_kwargs = {}

        x, y = self.xy

        artists: list[Artist] = [
            ax.scatter(
                x,
                y,
                *args,
                **kwargs,
            )
        ]

        if annotate:
            xytext: tuple[float, float] = self.xy + jnp.asarray(
                annotate_offset, dtype=float
            )  # type: ignore
            artists.append(
                ax.annotate(
                    annotate,
                    xy=(x, y),
                    xytext=(xytext[0], xytext[1]),
                    **annotate_kwargs,
                )
            )

        return artists

    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def bounding_box(self) -> Float[Array, "2 2"]:  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        return jnp.vstack([self.xy, self.xy])


@jaxtyped(typechecker=typechecker)
class Vertex(Point, Object, eqx.Module):
    """
    A vertex for corner diffraction.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Wall

        ax = plt.gca()
        wall = Wall(xys=jnp.array([[0., 0.], [1., 0.]]))
        _ = wall.plot(ax)
        for vertex in wall.get_vertices():
            _ = vertex.plot(ax)
        plt.show()  # doctest: +SKIP

    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def parameters_count() -> int:  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        return 0

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def parametric_to_cartesian(  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        self,
        param_coords: Float[Array, " {self.parameters_count()}"],  # type: ignore[reportUndefinedVariable]
    ) -> Float[Array, "2"]:
        return self.xy

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def cartesian_to_parametric(  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        self, carte_coords: Float[Array, "2"]
    ) -> Float[Array, " {self.parameters_count()}"]:  # type: ignore[reportUndefinedVariable]
        return jnp.empty_like(carte_coords, shape=0)

    @partial(jax.jit, static_argnames=("approx", "function"))
    @jaxtyped(typechecker=typechecker)
    def contains_parametric(  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        self,
        param_coords: Float[Array, " {self.parameters_count()}"],  # type: ignore[reportUndefinedVariable]
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Truthy:
        return true_value(approx=approx)

    @partial(jax.jit, static_argnames=("approx", "function"))
    @jaxtyped(typechecker=typechecker)
    def intersects_cartesian(  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        self,
        ray: Float[Array, "2 2"],
        patch: ScalarFloat = DEFAULT_PATCH,
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Truthy:
        return false_value(approx=approx)

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def evaluate_cartesian(self, ray_path: Float[Array, "3 2"]) -> Float[Array, " "]:  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        return jnp.array(0.0, dtype=ray_path.dtype)

    @jaxtyped(typechecker=typechecker)
    def plot(  # noqa: D102
        self, ax: Axes, *args: Any, **kwargs: Any
    ) -> MutableSequence[Artist]:  # pragma: no cover
        kwargs.setdefault("edgecolors", "black")
        kwargs.setdefault("facecolors", (1.0, 1.0, 0.0, 0.5))
        kwargs.setdefault("linestyle", "dashed")
        return super().plot(ax, *args, **kwargs)


@jaxtyped(typechecker=typechecker)
class Ray(Plottable, eqx.Module):
    """
    A ray object with origin and destination points.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Ray

        ax = plt.gca()
        ray = Ray(xys=jnp.array([[0., 0.], [1., 1.]]))
        _ = ray.plot(ax)
        plt.show()  # doctest: +SKIP

    """

    xys: Float[Array, "2 2"] = eqx.field(
        converter=jnp.asarray,
        default_factory=lambda: jnp.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    """Cartesian coordinates (origin, dest)."""

    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def origin(self) -> Float[Array, "2"]:
        """
        Returns the origin of this object.

        :return: The origin.
        """
        return self.xys[0, :]

    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def dest(self) -> Float[Array, "2"]:
        """
        Returns the destination of this object.

        :return: The destination.
        """
        return self.xys[1, :]

    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def t(self) -> Float[Array, "2"]:
        """
        Returns the direction vector of this object.

        :return: The direction vector.
        """
        return self.dest() - self.origin()

    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def rotate(
        self,
        angle: ScalarFloat,
        around: Optional[Union[Float[Array, "2"], Point]] = None,
    ) -> "Ray":
        """
        Returns a rotated copy of this ray.

        :param angle: The angle to rotate, in radian.
        :param around: An optional point to rotate around.
        :return: The rotated copy of this ray.

        :Examples:

        >>> from differt2d.geometry import Ray
        >>> import jax.numpy as jnp
        >>> ray = Ray(xys=jnp.array([[0.0, 0.0], [1.0, 0.0]]))
        >>> ray.rotate(angle=jnp.pi).xys
        Array([[ 8.742278e-08,  0.000000e+00],
               [-1.000000e+00,  0.000000e+00]], dtype=float32)
        """
        if around is None:
            center = jnp.array([0.0, 0.0])
        elif isinstance(around, Point):
            center = around.xy
        else:
            center = around

        c = jnp.cos(angle)
        s = jnp.sin(angle)

        rotation_matrix = jnp.array([[+c, -s], [+s, +c]])

        xys = self.xys - center[None, :]
        xys = rotation_matrix @ xys
        xys = xys + center[None, :]

        return eqx.tree_at(lambda ray: ray.xys, self, xys)

    @jaxtyped(typechecker=typechecker)
    def plot(self, ax: Axes, *args: Any, **kwargs: Any) -> MutableSequence[Artist]:  # noqa: D102
        kwargs.setdefault("color", "blue")
        x, y = self.xys.T
        return ax.plot(x, y, *args, **kwargs)  # type: ignore[func-returns-value]

    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def bounding_box(self) -> Float[Array, "2 2"]:  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        return jnp.vstack([jnp.min(self.xys, axis=0), jnp.max(self.xys, axis=0)])


@jaxtyped(typechecker=typechecker)
class Wall(Ray, Object, eqx.Module):
    """
    A wall object defined by its corners.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Wall

        ax = plt.gca()
        wall = Wall(xys=jnp.array([[0., 0.], [1., 0.]]))
        _ = wall.plot(ax)
        plt.show()  # doctest: +SKIP

    """

    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def normal(self) -> Float[Array, "2"]:
        """
        Returns the normal to the current wall, expressed in cartesian coordinates and normalized.

        :return: The normal.
        """
        t = self.t()
        n = t.at[0].set(t[1])
        n = n.at[1].set(-t[0])
        n, _ = normalize(n)
        return n

    @staticmethod
    @partial(jax.jit, inline=True)
    @jaxtyped(typechecker=typechecker)
    def parameters_count() -> int:  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        return 1

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def parametric_to_cartesian(  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        self,
        param_coords: Float[Array, " {self.parameters_count()}"],  # type: ignore[reportUndefinedVariable]
    ) -> Float[Array, "2"]:
        return self.origin() + param_coords * self.t()

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def cartesian_to_parametric(  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        self, carte_coords: Float[Array, "2"]
    ) -> Float[Array, " {self.parameters_count()}"]:  # type: ignore[reportUndefinedVariable]
        other = carte_coords - self.origin()
        squared_length = jnp.dot(self.t(), self.t())
        squared_length = jnp.where(squared_length == 0.0, 1.0, squared_length)
        return jnp.dot(self.t(), other).reshape(-1) / squared_length

    @partial(jax.jit, static_argnames=("approx", "function"))
    @jaxtyped(typechecker=typechecker)
    def contains_parametric(  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        self,
        param_coords: Float[Array, " {self.parameters_count()}"],  # type: ignore[reportUndefinedVariable]
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Truthy:
        param_coords = param_coords[0]
        ge = greater_equal(
            param_coords,
            jnp.array(0.0),
            approx=approx,
            **kwargs,
        )
        le = less_equal(
            param_coords,
            jnp.array(1.0),
            approx=approx,
            **kwargs,
        )
        return logical_and(ge, le, approx=approx)

    @partial(jax.jit, static_argnames=("approx", "function"))
    @jaxtyped(typechecker=typechecker)
    def intersects_cartesian(  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        self,
        ray: Float[Array, "2 2"],
        patch: ScalarFloat = DEFAULT_PATCH,
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Truthy:
        return segments_intersect(
            self.origin() - patch * self.t(),
            self.dest() + patch * self.t(),
            ray[0, :],
            ray[1, :],
            approx=approx,
            **kwargs,
        )

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def evaluate_cartesian(self, ray_path: Float[Array, "3 2"]) -> Float[Array, " "]:  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        i = ray_path[1, :] - ray_path[0, :]  # Incident
        r = ray_path[2, :] - ray_path[1, :]  # Reflected
        n = self.normal()  # Normal
        i, _ = normalize(i)
        r, _ = normalize(r)
        e = r - (i - 2 * jnp.dot(i, n) * n)
        return jnp.dot(e, e)

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def image_of(self, point: Float[Array, "2"]) -> Float[Array, "2"]:
        """
        Returns the image of a point with respect to this mirror (wall), using specular reflection.

        :param point: The starting point.
        :return: The image of the point.

        :Examples:

        >>> from differt2d.geometry import Wall
        >>> import jax.numpy as jnp
        >>> wall = Wall(xys=jnp.array([[0.0, 0.0], [1.0, 0.0]]))
        >>> wall.image_of(jnp.array([0.0, 1.0]))
        Array([ 0., -1.], dtype=float32)
        """
        i = point - self.origin()
        return point - 2.0 * jnp.dot(i, self.normal()) * self.normal()

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def get_vertices(self) -> tuple[Vertex, Vertex]:
        """
        Returns the two vertices of this wall.

        :return: The two vertices.
        """
        return Vertex(xy=self.xys[0, :]), Vertex(xy=self.xys[1, :])


@jaxtyped(typechecker=typechecker)
class RIS(Wall, eqx.Module):
    """
    A very basic Reflective Intelligent Surface (RIS) object.

    Here, we model a RIS such that the angle of reflection is constant
    with respect to its normal, regardless of the incident vector.
    """

    phi: Float[Array, " "] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.array(jnp.pi / 4)
    )
    """The constant angle of reflection."""

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def evaluate_cartesian(  # noqa: D102
        self, ray_path: Float[Array, "3 2"]
    ) -> Float[Array, " "]:
        r = ray_path[2, :] - ray_path[1, :]  # Reflected
        n = self.normal()  # Normal
        r, _ = normalize(r)
        sin_a = jnp.cross(-r, n)
        cos_a = jnp.dot(-r, n)
        sin_p = jnp.sin(self.phi)
        cos_p = jnp.cos(self.phi)
        return (sin_a - sin_p) ** 2 + (cos_a - cos_p) ** 2

    @jaxtyped(typechecker=typechecker)
    def plot(  # noqa: D102
        self, ax: Axes, *args: Any, **kwargs: Any
    ) -> MutableSequence[Artist]:  # pragma: no cover
        kwargs.setdefault("color", "green")
        return super().plot(ax, *args, **kwargs)


@jaxtyped(typechecker=typechecker)
class Path(Plottable, eqx.Module):
    """
    A path object with at least two points.

    .. plot::
        :include-source: true

        import matplotlib.pyplot as plt
        import jax.numpy as jnp
        from differt2d.geometry import Path

        ax = plt.gca()
        path = Path(xys=jnp.array([[0., 0.], [.8, .2], [1., 1.]]))
        _ = path.plot(ax)
        plt.show()  # doctest: +SKIP

    """

    xys: Float[Array, "num_points 2"] = eqx.field(converter=jnp.asarray)
    """Array of cartesian coordinates."""

    loss: Float[Array, " "] = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.array(0.0)
    )
    """The loss value for the given path."""

    @classmethod
    @jaxtyped(typechecker=typechecker)
    def from_tx_objects_rx(
        cls,
        tx: Union[Float[Array, "2"], Point],
        objects: Sequence[Interactable],
        rx: Union[Float[Array, "2"], Point],
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs: Any,
    ) -> "Path":
        """
        Returns a path from TX to RX, traversing each object in the list in the provided order.

        The present implementation will sample a point at :python:`t = 0.5`
        on each object.

        :param tx: The transmitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :param key: Unused for this class,
            kept for compatibility with other path classes.
        :param kwargs: Unused for this class,
            kept for compatibility with other path classes.
        :return: The resulting path.

        :Examples:

        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            import jax.numpy as jnp
            from differt2d.geometry import Path
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            path = Path.from_tx_objects_rx(
                scene.transmitters["tx"],
                scene.objects,
                scene.receivers["rx"]
            )
            _ = path.plot(ax)
            plt.show()  # doctest: +SKIP
        """
        if isinstance(tx, Point):
            tx = tx.xy

        if isinstance(rx, Point):
            rx = rx.xy

        xys = [obj.parametric_to_cartesian(jnp.array([0.5])) for obj in objects]
        xys = jnp.vstack([tx, *xys, rx])
        return cls(xys=xys)

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def length(self) -> Float[Array, " "]:
        """
        Returns the length of this path.

        :return: The path length.
        """
        return path_length(self.xys)

    @partial(jax.jit, static_argnames=("approx", "function"))
    @jaxtyped(typechecker=typechecker)
    def on_objects(
        self,
        objects: Sequence[Interactable],
        approx: Optional[bool] = None,
        **kwargs,
    ) -> Truthy:
        """
        Returns whether the path correctly passes on the objects.

        For each object i, it will check whether it contains the ith
        point in the path (start and end points are ignored).

        :param objects: The list of objects to check against.
        :param approx: Whether approximation is enabled or not.
        :param kwargs: Keyword arguments passed to
            :func:`activation<differt2d.logic.activation>`.
        :return: Whether this path passes on the objects.
        """
        contains = true_value(approx=approx)
        for i, obj in enumerate(objects):
            param_coords = obj.cartesian_to_parametric(self.xys[i + 1, :])
            contains = logical_and(
                contains,
                obj.contains_parametric(
                    param_coords,
                    approx=approx,
                    **kwargs,
                ),
                approx=approx,
            )

        return contains

    @partial(jax.jit, inline=True, static_argnames=("approx", "function"))
    @jaxtyped(typechecker=typechecker)
    def intersects_with_objects(
        self,
        objects: Sequence[Interactable],
        path_candidate: Array,
        patch: ScalarFloat = DEFAULT_PATCH,
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Truthy:
        """
        Returns whether the path intersects with any of the objects.

        The ``path_candidate`` argument is used to avoid checking for
        intersection on objects the path is allowed to pass on.

        :param objects: The list of objects in the scene.
        :param path_candidate: The object indices on which the path should pass.
        :param patch: The patch value for intersection check,
            see :meth:`Interactable.intersects_cartesian<differt2d.abc.Interactable.intersects_cartesian>`.
        :param approx: Whether approximation is enabled or not.
        :param kwargs: Keyword arguments passed to
            :func:`activation<differt2d.logic.activation>`.
        :return: Whether this path intersects any of the objects.
        """
        interacting_object_indices = [-1] + [i for i in path_candidate] + [-1]
        intersects = false_value(approx=approx)

        for i in range(self.xys.shape[0] - 1):
            ray_path = self.xys[i : i + 2, :]
            for obj_index, obj in enumerate(objects):
                ignore = jnp.logical_or(
                    obj_index == interacting_object_indices[i + 0],
                    obj_index == interacting_object_indices[i + 1],
                )
                intersects = jnp.where(
                    ignore,
                    intersects,
                    logical_or(
                        intersects,
                        obj.intersects_cartesian(
                            ray_path,
                            patch=patch,
                            approx=approx,
                            **kwargs,
                        ),
                        approx=approx,
                    ),
                )

        return intersects

    @partial(jax.jit, inline=True, static_argnames=("approx", "function"))
    @jaxtyped(typechecker=typechecker)
    def is_valid(
        self,
        objects: Sequence[Interactable],
        path_candidate: Int[Array, " order"],
        interacting_objects: Sequence[Interactable],
        tol: ScalarFloat = 1e-2,
        patch: ScalarFloat = DEFAULT_PATCH,
        approx: Optional[bool] = None,
        **kwargs: Any,
    ) -> Truthy:
        """
        Returns whether the current path is valid, according to three requirements (see below).

        The requirements are:

        1. the path loss is below some tolerance;
        2. the coordinate points are correctly placed inside the corresponding
           interacting objects;
        3. and the path does not intersect with any of the objects in the scene
           (except those concerned by 2.).

        :param objects: The objects in the scene.
        :param path_candidate: The array of indices of interacting objects,
            usually generated with
            :meth:`Scene.all_path_candidates<differt2d.scene.Scene.all_path_candidates>`
        :param interacting_objects: The list of interacting objects,
            usually obtained by calling
            :meth:`Scene.get_interacting_objects<differt2d.scene.Scene.get_interacting_objects>`.
        :param tol: The maximum allowed value for the path loss before it is considered
            invalid. I.e., a path loss greater than ``tol`` will make this path invalid.
        :param patch: The patch value for intersection check,
            see :meth:`Interactable.intersects_cartesian<differt2d.abc.Interactable.intersects_cartesian>`.
        :param approx: Whether approximation is enabled or not.
        :param kwargs: Keyword arguments passed to
            :func:`activation<differt2d.logic.activation>`.
        :return: Whether this path is valid.
        """
        return jnp.nan_to_num(
            logical_all(
                self.on_objects(interacting_objects, approx=approx, **kwargs),
                logical_not(
                    self.intersects_with_objects(
                        objects,
                        path_candidate,
                        patch=patch,
                        approx=approx,
                        **kwargs,
                    ),
                    approx=approx,
                ),
                less(self.loss, jnp.asarray(tol), approx=approx, **kwargs),
                approx=approx,
            )
        )

    @jaxtyped(typechecker=typechecker)
    def plot(self, ax: Axes, *args: Any, **kwargs: Any) -> MutableSequence[Artist]:  # noqa: D102
        kwargs.setdefault("color", "orange")
        x, y = self.xys.T
        return ax.plot(x, y, *args, **kwargs)  # type: ignore

    @jaxtyped(typechecker=typechecker)
    def bounding_box(self) -> Float[Array, "2 2"]:  # type: ignore[reportIncompatibleMethodOverride] # noqa: D102
        return jnp.vstack([jnp.min(self.xys, axis=0), jnp.max(self.xys, axis=0)])


@partial(jax.jit, inline=True, static_argnames=("size",))
@jaxtyped(typechecker=typechecker)
def parametric_to_cartesian_from_slice(  # noqa: D103
    obj: Interactable,
    parametric_coords: Float[Array, " num_parametric_coords"],
    start: ScalarInt,
    size: ScalarInt,
) -> Float[Array, "2"]:
    parametric_coords = jax.lax.dynamic_slice(parametric_coords, (start,), (size,))
    return obj.parametric_to_cartesian(parametric_coords)


@partial(jax.jit, inline=True, static_argnames=("n",))
@jaxtyped(typechecker=typechecker)
def parametric_to_cartesian(  # noqa: D103
    objects: Sequence[Interactable],
    parametric_coords: Float[Array, " num_parametric_coords"],
    n: int,
    tx_coords: Float[Array, "2"],
    rx_coords: Float[Array, "2"],
) -> Float[Array, "{n}+2 2"]:
    assert n >= 0, f"n must be greater than 0, got {n = } instead."

    cartesian_coords = jnp.empty((n + 2, 2))
    cartesian_coords = cartesian_coords.at[0].set(tx_coords)
    cartesian_coords = cartesian_coords.at[-1].set(rx_coords)
    j = jnp.int32(0)
    for i, obj in enumerate(objects):
        size = obj.parameters_count()
        cartesian_coords = cartesian_coords.at[i + 1].set(
            parametric_to_cartesian_from_slice(obj, parametric_coords, j, size)
        )
        j += size

    return cartesian_coords


@jaxtyped(typechecker=typechecker)
class ImagePath(Path, eqx.Module):
    """A path object that was obtained with the Image method."""

    @classmethod
    @partial(jax.jit, static_argnames=("cls",))
    @jaxtyped(typechecker=typechecker)
    def from_tx_objects_rx(  # type: ignore
        cls,
        tx: Union[Float[Array, "2"], Point],
        objects: Sequence[Wall],
        rx: Union[Float[Array, "2"], Point],
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs: Any,
    ) -> "ImagePath":
        """
        Returns a path with minimal length.

        :param tx: The transmitting node.
        :param objects:
            The list of walls to interact with (order is important).
        :param rx: The receiving node.
        :param key: Unused for this class,
            kept for compatibility with other path classes.
        :param kwargs: Unused for this class,
            kept for compatibility with other path classes.
        :return: The resulting path of the Image method.

        :Examples:

        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            import jax.numpy as jnp
            from differt2d.geometry import ImagePath
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            path = ImagePath.from_tx_objects_rx(
                scene.transmitters["tx"],
                scene.objects,
                scene.receivers["rx"]
            )
            _ = path.plot(ax)
            plt.show()  # doctest: +SKIP
        """
        n = len(objects)

        if isinstance(tx, Point):
            tx = tx.xy

        if isinstance(rx, Point):
            rx = rx.xy

        if n == 0:
            xys = jnp.vstack([tx, rx])
            return cls(xys=xys, loss=jnp.array(0.0))

        walls = stack_leaves(objects)

        @jax.jit
        @jaxtyped(typechecker=typechecker)
        def path_loss(
            cartesian_coords: Float[Array, "path_length 2"],
        ) -> Float[Array, " "]:
            _loss = jnp.array(0.0)
            for i, obj in enumerate(objects):
                _loss += obj.evaluate_cartesian(cartesian_coords[i : i + 3, :])

            return _loss

        @jaxtyped(typechecker=typechecker)
        def forward(
            image: Float[Array, "2"], wall: Wall
        ) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
            image = wall.image_of(image)
            return image, image

        @jaxtyped(typechecker=typechecker)
        def backward(
            point: Float[Array, "2"], x: tuple[Wall, Float[Array, "2"]]
        ) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
            wall, image = x
            p = wall.origin()
            n = wall.normal()
            u = point - image
            v = p - point
            un = jnp.dot(u, n)
            vn = jnp.dot(v, n)
            # Avoid division by zero
            inc = jnp.where(un == 0.0, 0.0, vn * u / un)
            point = point + inc
            return point, point

        _, images = jax.lax.scan(forward, init=tx, xs=walls)  # type: ignore
        _, points = jax.lax.scan(backward, init=rx, xs=(walls, images), reverse=True)  # type: ignore

        xys = jnp.vstack([tx, *points, rx])

        return cls(xys=xys, loss=path_loss(xys))


@jaxtyped(typechecker=typechecker)
class FermatPath(Path, eqx.Module):
    """A path object that was obtained with the Fermat's Principle Tracing method."""

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "steps", "many", "optimizer"))
    @jaxtyped(typechecker=None)
    def from_tx_objects_rx(  # type: ignore
        cls,
        tx: Union[Float[Array, "2"], Point],
        objects: Sequence[Interactable],
        rx: Union[Float[Array, "2"], Point],
        *,
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> "FermatPath":
        """
        Returns a path with minimal length.

        :param tx: The transmitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :param key: The random key to generate the initial guess.
        :param kwargs:
            Keyword arguments passed to
            :func:`minimize_many_random_uniform<differt2d.optimize.minimize_many_random_uniform>`.
            Note that the ``many`` parameter defaults to ``1`` here.
        :return: The resulting path of the FPT method.

        :Examples:

        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            import jax
            import jax.numpy as jnp
            from differt2d.geometry import FermatPath
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            path = FermatPath.from_tx_objects_rx(
                scene.transmitters["tx"],
                scene.objects,
                scene.receivers["rx"],
                key=jax.random.PRNGKey(1234),
            )
            _ = path.plot(ax)
            plt.show()  # doctest: +SKIP
        """
        n = len(objects)

        if isinstance(tx, Point):
            tx = tx.xy

        if isinstance(rx, Point):
            rx = rx.xy

        if n == 0:
            xys = jnp.vstack([tx, rx])
            return cls(xys=xys, loss=jnp.array(0.0))

        n_unknowns = sum([obj.parameters_count() for obj in objects])

        @jax.jit
        @jaxtyped(typechecker=typechecker)
        def loss_fun(theta: Float[Array, " n_unknowns"]) -> Float[Array, " "]:
            cartesian_coords = parametric_to_cartesian(objects, theta, n, tx, rx)

            return path_length(cartesian_coords)

        @jax.jit
        @jaxtyped(typechecker=typechecker)
        def path_loss(
            cartesian_coords: Float[Array, " n_cartesian_unknowns 2"],
        ) -> Float[Array, " "]:
            _loss = jnp.array(0.0)
            for i, obj in enumerate(objects):
                _loss += obj.evaluate_cartesian(cartesian_coords[i : i + 3, :])

            return _loss

        kwargs.setdefault("many", 1)

        theta, _ = minimize_many_random_uniform(
            fun=loss_fun, n=n_unknowns, key=key, **kwargs
        )

        xys = parametric_to_cartesian(objects, theta, n, tx, rx)

        return cls(xys=xys, loss=path_loss(xys))


@jaxtyped(typechecker=typechecker)
class MinPath(Path, eqx.Module):
    """A path object that was obtained with the Min-Path-Tracing method :cite:`mpt-eucap2023`."""

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "steps", "many", "optimizer"))
    @jaxtyped(typechecker=None)
    def from_tx_objects_rx(  # type: ignore
        cls,
        tx: Union[Float[Array, "2"], Point],
        objects: Sequence[Interactable],
        rx: Union[Float[Array, "2"], Point],
        *,
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> "MinPath":
        """
        Returns a path that minimizes the sum of interactions.

        :param tx: The transmitting node.
        :param objects:
            The list of objects to interact with (order is important).
        :param rx: The receiving node.
        :param key: The random key to generate the initial guess.
        :param kwargs:
            Keyword arguments passed to
            :func:`minimize_many_random_uniform<differt2d.optimize.minimize_many_random_uniform>`.
            Note that the ``many`` parameter defaults to ``1`` here.
        :return: The resulting path of the MPT method.

        :Examples:

        .. plot::
            :include-source: true

            import matplotlib.pyplot as plt
            import jax
            import jax.numpy as jnp
            from differt2d.geometry import MinPath
            from differt2d.scene import Scene

            ax = plt.gca()
            scene = Scene.square_scene()
            _ = scene.plot(ax)
            path = MinPath.from_tx_objects_rx(
                scene.transmitters["tx"],
                scene.objects,
                scene.receivers["rx"],
                key=jax.random.PRNGKey(1234),
            )
            _ = path.plot(ax)
            plt.show()  # doctest: +SKIP
        """
        n = len(objects)

        if isinstance(tx, Point):
            tx = tx.xy

        if isinstance(rx, Point):
            rx = rx.xy

        if n == 0:
            xys = jnp.vstack([tx, rx])
            return cls(xys=xys, loss=jnp.array(0.0))

        n_unknowns = sum(obj.parameters_count() for obj in objects)

        @jax.jit
        @jaxtyped(typechecker=typechecker)
        def loss_fun(theta: Float[Array, " n_unknowns"]) -> Float[Array, " "]:
            cartesian_coords = parametric_to_cartesian(objects, theta, n, tx, rx)
            _loss = jnp.array(0.0)
            for i, obj in enumerate(objects):
                _loss += obj.evaluate_cartesian(cartesian_coords[i : i + 3, :])

            return _loss

        kwargs.setdefault("many", 1)

        theta, loss = minimize_many_random_uniform(
            fun=loss_fun, n=n_unknowns, key=key, **kwargs
        )

        xys = parametric_to_cartesian(objects, theta, n, tx, rx)

        return cls(xys=xys, loss=loss)
