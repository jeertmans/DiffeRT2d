"""
Animate power optimization using approximation
==============================================

This example shows how one can use approximation to perform
power optimization on a given network configuration.

Here, to goal is to find the transmitter location that
maximizes some ``objective_function``, using the approximation framework
and the :func:`optax.adam` optimizer.

To reach a realistic optimum, approximation's ``alpha`` value is increased
step after step, using a geometric progression.
"""

# %%
# Imports
# -------
#
# First, we need to import the necessary modules.

from collections.abc import Iterator
from copy import copy
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

from differt2d.geometry import Point
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

# %%
# Scene
# -----
#
# The following code will work with any scene, but be aware that large scenes
# may introduces a long computation time.
#
# You can easily change the scene by modifying the following line:

scene = Scene.square_scene_with_obstacle()

# %%
# Defining an objective function
# ------------------------------
#
# In an optimization problem, one must first define an objective function.
# Ideally, in a telecommunications scenario, we would like to serve all users,
# i.e., receivers, with a good power. Because we want all users to receive the
# maximum power possible, we will maximize the minimum received power among all
# users.
#
# Finally, we define a loss function that will take the opposite of the
# objective function, because most optimizer minimize functions.


def objective_function(
    received_power_per_receiver: Iterator[Float[Array, " *batch"]],
) -> Float[Array, " *batch"]:
    """
    Objective function, that wants to maximize the received power by each
    receiver.
    """
    acc = jnp.array(jnp.inf)
    for p in received_power_per_receiver:
        p = p / P0  # Normalize power
        acc = jnp.minimum(acc, p)

    return acc


def loss(
    tx_coords: Float[Array, "2"], scene: Scene, *args: Any, **kwargs: Any
) -> Float[Array, " "]:
    """Loss function, to be minimized."""
    scene = scene.with_transmitters(tx=Point(xy=tx_coords))
    return -objective_function(
        power for _, _, power in scene.accumulate_over_paths(*args, **kwargs)
    )


f_and_df = jax.value_and_grad(
    loss
)  # Generates a function that evaluates f and its gradient

# %%
# Plot setup
# ----------
#
# Below, we setup the plot for animation.
#
# .. note::
#
#    The transmitter is intentionally placed in a zero-gradient zone, to showcase
#    the problem of non-convergence when not using approximation. Note that
#    the zero gradient region can be avoided if one simulates a ``max_order``
#    greater than ``0`` (e.g., ``1`` is sufficient here). This, of course,
#    depends on the scene.

fig, axes = plt.subplots(2, 1, sharex=True, tight_layout=True)

annotate_kwargs = dict(color="red", fontsize=12, fontweight="bold")

scene = scene.with_transmitters(tx=Point(xy=jnp.array([0.5, 0.7])))
scene = scene.with_receivers(
    rx_0=Point(xy=jnp.array([0.3, 0.1])), rx_1=Point(xy=jnp.array([0.5, 0.1]))
)

X, Y = scene.grid(n=300)

im_artists = []
transmitter_artists = []
annotate_artists = []
scenes = [scene, copy(scene)]  # Need a copy, because scenes will diverge

for ax, approx, scene in zip(axes, [False, True], scenes):
    scene_artists = scene.plot(
        ax,
        transmitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
        receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
    )
    transmitter_artists.append(scene_artists[0])
    annotate_artists.append(scene_artists[1])

    im = ax.pcolormesh(
        X, Y, jnp.ones_like(X), norm=LogNorm(vmin=1e-4, vmax=1e0), zorder=-1
    )
    im_artists.append(im)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Objective function")
    ax.set_ylabel("y coordinate")
    ax.set_title("With approximation" if approx else "Without approximation")

axes[-1].set_xlabel("x coordinate")

steps = 101  # In how many steps we hope to converge

# sphinx_gallery_defer_figures

# %%
# Choosing the right alpha values
# -------------------------------
#
# Theoretically, one should choose an infinitely big ``alpha`` value
# to reduce the approximation to zero.
#
# However, it can be observed that values above 100.0 are already high enough
# to damp any approximation effect.
#
# A basic optimizer is used, but your are encouraged to test various
# ``alpha`` progressions and optimizers.

alphas = jnp.logspace(0, 2, steps)  # Values between 1.0 and 100.0

# Dummy values, to be filled by ``init_func``.
optimizers: list[optax.GradientTransformation] = [None, None]  # type: ignore
carries: list[tuple[Float[Array, "2"], optax.OptState]] = [(None, None), (None, None)]  # type: ignore

# sphinx_gallery_defer_figures

# %%
# Animation functions
# -------------------
#
# As required by :class:`maplotlib.animations.FuncAnimation`, we must
# define a function that will be called on every animation frame.
#
# Here, we choose to play one optimization step per frame, and the pass
# the corresponding ``alpha`` values directly as an argument.


def init_func() -> list:
    tx_coords = jnp.array([0.5, 0.7])
    for i in range(2):
        scenes[i] = scene.with_transmitters(tx=Point(xy=tx_coords))
        optimizers[i] = optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans())
        carries[i] = tx_coords, optimizers[i].init(tx_coords)

    return []


def func(alpha: float) -> list:
    for i, approx in enumerate([False, True]):
        tx_coords, opt_state = carries[i]

        # Plotting prior to updating
        scenes[i] = scene.with_transmitters(tx=Point(xy=tx_coords))
        transmitter_artists[i].set_offsets([tx_coords[0], tx_coords[1]])
        annotate_artists[i].set_x(tx_coords[0])
        annotate_artists[i].set_y(tx_coords[1])

        loss, grads = f_and_df(
            tx_coords,
            scenes[i],
            fun=received_power,
            max_order=0,
            approx=approx,
            alpha=alpha,
        )

        F = objective_function(
            power  # type: ignore
            for _, power in scenes[i].accumulate_on_transmitters_grid_over_paths(
                X, Y, fun=received_power, max_order=0, approx=approx, alpha=alpha
            )
        )
        im_artists[i].set_array(F)

        updates, opt_state = optimizers[i].update(grads, opt_state)
        tx_coords = tx_coords + updates

        carries[i] = tx_coords, opt_state

        if approx:
            alpha_str = f"{alpha:.2e}"
            base, expo = alpha_str.split("e")
            expo = str(int(expo))  # Remove trailing zeros and +
            axes[i].set_title(f"With approximation - $\\alpha={alpha:.2e}$")

    return []


anim = FuncAnimation(fig, func=func, init_func=init_func, frames=alphas, interval=100)
plt.show()  # doctest: +SKIP
