"""
Plot received power over a grid
===============================

This examples shows how one can plot the received power map, i.e.,
the power received at each (x, y) coordinate as the sum of the power from
each emitter.

The first plot shows the power map when no approximation is used, and the
second shows the power map, but when approximation is enabled (with default parameters,
see :mod:`defaults<differt2d.defaults>`).
"""

# %%
# Imports
# -------
#
# First, we need to import the necessary modules.

import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array

from differt2d.scene import Scene
from differt2d.utils import P0, received_power, setup_fig_for_paper

scene = Scene.square_scene_with_wall()

# %%
# Scene
# -----
#
# The following code will work with any scene, but be aware that large scenes
# may introduces a long computation time.
#
# You can easily change the scene by modifying the following line:

# %%
# Plot setup
# ----------
#
# Below, we setup the plot to contain two axes: one showing the power map
# without approximation, and one showing the power map using approximation.

fig, axes = plt.subplots(2, 1, sharex=True, tight_layout=True)
setup_fig_for_paper(fig)

annotate_kwargs = dict(color="white", fontsize=12, fontweight="bold")

X, Y = scene.grid(n=600)

for ax, approx in zip(axes, [False, True]):
    scene.plot(
        ax,
        emitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
        receivers=False,
    )

    P: Array = scene.accumulate_on_receivers_grid_over_paths(
        X, Y, fun=received_power, reduce=True, approx=approx
    )

    PdB = 10.0 * jnp.log10(P / P0)

    im = ax.pcolormesh(X, Y, PdB, vmin=-50, vmax=5, rasterized=True, zorder=-1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Power (dB)")

    ax.set_ylabel("y coordinate")
    ax.set_title("With approximation" if approx else "Without approximation")

axes[-1].set_xlabel("x coordinate")
fig.savefig("power_map.pgf", dpi=300)
# plt.show()