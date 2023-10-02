"""
Animate power optimization using approximation
==============================================

This example shows how one can use approximation to perform
power optimization on a given network configuration.

Here, to goal is to find the emitter location that
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

from copy import deepcopy as copy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matplotlib.animation import FuncAnimation

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
# i.e., receivers, with a good power. Because antennas are often designed to work
# with power expressed in a logarithmic scale, our objective function will
# reflect that and sum the received power by each user separately, in dB.
#
# Finally, we define a loss function that will take the opposite of the
# objective function, because most optimizer minimize functions.


def objective_function(received_power_per_receiver):
    """Objective function, that wants to maximise the sum of receiver power by each user
    (in dB)."""
    acc = 0.0
    for p in received_power_per_receiver:
        p = p / P0  # Normalize power
        acc = acc + jnp.log10(p)

    return 10.0 * acc


def loss(tx_coords, scene, *args, **kwargs):
    """Loss function, to be minimized."""
    scene.emitters["tx"].point = tx_coords
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
#    The emitter is intentionnally placed in a zero-gradient zone, to showcase
#    the problem of non-convergence when not using approximation.

fig, axes = plt.subplots(2, 1, sharex=True)

annotate_kwargs = dict(color="red", fontsize=12, fontweight="bold")

scene.emitters = dict(
    tx=Point(point=jnp.array([0.5, 0.7])),
)
scene.receivers = dict(
    tx_0=Point(point=jnp.array([0.3, 0.1])),
    tx_1=Point(point=jnp.array([0.5, 0.1])),
)

X, Y = scene.grid(n=300)

im_artists = []
emitter_artists = []
annotate_artists = []
scenes = [scene, copy(scene)]  # Need a copy, because scenes will diverge

for ax, approx, scene in zip(axes, [False, True], scenes):
    scene_artists = scene.plot(
        ax,
        emitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
        receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
    )
    emitter_artists.append(scene_artists[0])
    annotate_artists.append(scene_artists[1])

    im = ax.pcolormesh(X, Y, jnp.zeros_like(X), vmin=-60, vmax=5, zorder=-1)
    im_artists.append(im)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Objective function")
    ax.set_ylabel("y coordinate")
    ax.set_title("With approximation" if approx else "Without approximation")

axes[-1].set_xlabel("x coordinate")

steps = 100  # In how many steps we hope to converge

# sphinx_gallery_defer_figures

# %%
# Choosing the right alpha values
# -------------------------------
#
# Theoritically, one should choose an infinitely big ``alpha`` value
# to reduce the approximation to zero.
#
# However, it can be observed that values above 100.0 are already high enough
# to damp any approximation effect.
#
# A basic optimizer is used, but your are encouraged to test various
# ``alpha`` progressions and optimizers.

alphas = jnp.logspace(0, 2, steps)  # Values between 1.0 and 100.0

# Dummy values, to be filled by ``init_func``.
optimizers = [None, None]
carries = [(None, None), (None, None)]


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


def init_func():
    tx_coords = jnp.array([0.5, 0.7])
    for i, scene in enumerate(scenes):
        scene.emitters["tx"].point = tx_coords
        optimizers[i] = optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans())
        carries[i] = tx_coords, optimizers[i].init(tx_coords)


def func(alpha):
    for i, approx in enumerate([False, True]):
        tx_coords, opt_state = carries[i]
        loss, grads = f_and_df(
            tx_coords,
            scenes[i],
            fun=received_power,
            max_order=0,
            approx=approx,
            alpha=alpha,
        )
        updates, opt_state = optimizers[i].update(grads, opt_state)
        tx_coords = tx_coords + updates

        carries[i] = tx_coords, opt_state
        scenes[i].emitters["tx"].point = tx_coords
        emitter_artists[i].set_data([tx_coords[0]], [tx_coords[1]])
        annotate_artists[i].set_x(tx_coords[0])
        annotate_artists[i].set_y(tx_coords[1])

        F = objective_function(
            power
            for _, power in scenes[i].accumulate_on_emitters_grid_over_paths(
                X, Y, fun=received_power, max_order=0, approx=approx, alpha=alpha
            )
        )
        im_artists[i].set_array(F)

        if approx:
            axes[i].set_title(f"With approximation - $\\alpha={alpha:.2e}$")


anim = FuncAnimation(fig, func=func, init_func=init_func, frames=alphas, interval=100)
plt.tight_layout()
plt.show()
