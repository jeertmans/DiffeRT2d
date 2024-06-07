"""
Plot power profiles for various parameters
==========================================

This example shows how one can plot the received power profiles, i.e.,
the power received along a line as the sum of the power from
each transmitter, for a variety of approximation parameters.

The receiver shown on the plot is just indicative, but are not actually used
in the process of computing the power profiles.

Various lines shows how the ``alpha`` parameter
in :func:`activation<differt2d.logic.activation>` impacts the power computation.


As shown, the higher ``alpha``, the closer it gets to the `Without approx.` case.
"""

# %%
# Imports
# -------
#
# First, we need to import the necessary modules.

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float

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

scene = Scene.square_scene_with_wall()

# %%
# Plot setup
# ----------
#
# Below, we setup the plot with two axes.

fig, axes = plt.subplots(2, 1, sharex=True, tight_layout=True)

annotate_kwargs = dict(color="red", fontsize=12, fontweight="bold")

# sphinx_gallery_defer_figures

# %%
# First axis
# ^^^^^^^^^^
#
# On the first axis, we plot a top-down view of the scene, as well as
# the power map for a fixed transmitter location, and a received location
# given by the coordinates.
# This is performed without approximation.

scene.plot(
    axes[0],
    transmitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
    receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
)

X, Y = scene.grid(n=300)
P: Float[Array, "300 300"] = scene.accumulate_on_receivers_grid_over_paths(
    X,
    Y,
    fun=received_power,
    reduce_all=True,
    approx=False,
    min_order=0,
    max_order=0,
)  # type: ignore

PdB = 10.0 * jnp.log10(P / P0)

axes[0].pcolormesh(X, Y, PdB, vmin=-50, vmax=5, zorder=-1)
axes[0].set_ylabel("y coordinate")
_ = axes[1].set_title("Without approx.")  # dummy assign only needed for docs

# sphinx_gallery_defer_figures

# %%
# Second axis
# ^^^^^^^^^^^
#
# We plot various power profiles, along a line that joins the transmitter and
# receiver shown on the first axis.
#
# The first profile is the no-approximation case. Subsequent profiles are
# using approximation and have different ``alpha`` values.

x = jnp.linspace(0.2, 0.8, 200)
y = jnp.array([0.5])

X, Y = jnp.meshgrid(x, y)

P: Float[Array, "200 1"] = scene.accumulate_on_receivers_grid_over_paths(
    X, Y, fun=received_power, reduce_all=True, approx=False, min_order=0, max_order=0
)  # type: ignore

PdB = 10.0 * jnp.log10(P.reshape(-1) / P0)

axes[1].plot(x, PdB, label="Without")

for alpha in [1.0, 10.0, 100.0, 1000.0]:
    P: Float[Array, "200 1"] = scene.accumulate_on_receivers_grid_over_paths(
        X,
        Y,
        fun=received_power,
        reduce_all=True,
        approx=True,
        alpha=alpha,
        min_order=0,
        max_order=0,
    )  # type: ignore

    PdB = 10.0 * jnp.log10(P.reshape(-1) / P0)

    axes[1].plot(x, PdB, label=f"With + $\\alpha = {alpha:.0e}$")

axes[1].set_ylabel("Power (dB)")
axes[1].set_title("With approx.")
axes[1].set_ylim([-20, 0])

axes[-1].set_xlabel("x coordinate")
plt.legend()
plt.show()  # doctest: +SKIP
