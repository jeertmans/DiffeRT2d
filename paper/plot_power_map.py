from pathlib import Path

import jax.numpy as jnp
from chex import Array
from utils import create_fig_for_paper

from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_wall()

fig, axes = create_fig_for_paper(2, 1, sharex=True, tight_layout=True)

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

folder = Path(__file__).parent / "pgf"
folder.mkdir(exist_ok=True)
fig.savefig(folder / "power_map.pgf", dpi=300)
