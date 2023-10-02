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
from differt2d.utils import P0, received_power, setup_fig_for_paper

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

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, tight_layout=True)
setup_fig_for_paper(fig, columns=2, height_to_width_ratio=1.0/3.0)

annotate_kwargs = dict(color="red", fontsize=12, fontweight="bold")

scene.emitters = dict(
    tx=Point(point=jnp.array([0.5, 0.7])),
)
scene.receivers = {
    r"rx\_0": Point(point=jnp.array([0.3, 0.1])),
    r"rx\_1": Point(point=jnp.array([0.5, 0.1])),
}

X, Y = scene.grid(n=600)

scenes = [scene, copy(scene)]  # Need a copy, because scenes will diverge

steps = 80  # In how many steps we hope to converge

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


tx_coords = jnp.array([0.5, 0.7])
for i, scene in enumerate(scenes):
    scene.emitters["tx"].point = tx_coords
    optimizers[i] = optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans())
    carries[i] = tx_coords, optimizers[i].init(tx_coords)

for frame, alpha in enumerate(alphas):

    for i, approx in enumerate([False, True]):
        tx_coords, opt_state = carries[i]
        scenes[i].emitters["tx"].point = tx_coords

        # Plotting prior to updating
        if frame % 20 == 0 and approx:
            ax = axes.flat[frame // 20]

            scene_artists = scenes[i].plot(
                ax,
                emitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
                receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
            )

            F = objective_function(
                power
                for _, power in scenes[i].accumulate_on_emitters_grid_over_paths(
                    X, Y, fun=received_power, max_order=0, approx=approx, alpha=alpha
                )
            )

            im = ax.pcolormesh(
                X, Y, F, vmin=-60, vmax=5, zorder=-1, rasterized=True
            )
            if (frame // 20) == 0:
                ax.set_ylabel("y coordinate")

            if i == 1:
                ax.set_xlabel("x coordinate")

            if approx:
                alpha_str = f"{alpha:.2e}"
                base, expo = alpha_str.split("e")
                expo = str(int(expo))  # Remove trailing zeros and +
                ax.set_title(
                    r"$\alpha="
                    + base[:-1]
                    + r"\times 10^{"
                    + expo
                    + "}$"
                )

        # Perform optimizer update
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


fig.savefig("mdr.pdf", dpi=300)
