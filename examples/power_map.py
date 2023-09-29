"""
Plot power over a grid.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array

from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_wall()

fig, axes = plt.subplots(2, 1, sharex=True)

annotate_kwargs = dict(color="white", fontsize=12, fontweight="bold")

X, Y = scene.grid(n=300)

for ax, approx in zip(axes, [False, True]):
    scene.plot(
        ax,
        emitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
        receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
    )

    P: Array = scene.accumulate_on_receivers_grid_over_paths(
        X, Y, fun=received_power, reduce=True, approx=approx
    )

    PdB = 10.0 * jnp.log10(P / P0)

    im = ax.pcolormesh(X, Y, PdB, vmin=-50, vmax=5, zorder=-1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Power (dB)")

    ax.set_ylabel("y coordinate")
    ax.set_title("With approximation" if approx else "Without approximation")

axes[-1].set_xlabel("x coordinate")
plt.show()
