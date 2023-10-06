from pathlib import Path

import jax.numpy as jnp
from chex import Array
from matplotlib.colors import LogNorm
from utils import create_fig_for_paper

from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_wall()
scene.emitters["TX"] = scene.emitters.pop("tx")

annotate_kwargs = dict(color="black", fontsize=10, fontweight="bold", ha="center")
point_kwargs = dict(
    markersize=3, annotate_offset=(0, 0.05), annotate_kwargs=annotate_kwargs
)

X, Y = scene.grid(n=600)

for grad in [False, True]:
    fig, axes = create_fig_for_paper(2, 1, sharex=True, tight_layout=True)
    for ax, approx in zip(axes, [False, True]):
        scene.plot(
            ax,
            emitters_kwargs=point_kwargs,
            receivers=False,
        )

        P: Array = scene.accumulate_on_receivers_grid_over_paths(
            X, Y, fun=received_power, reduce_all=True, grad=grad, approx=approx
        )

        if grad:
            dP = jnp.linalg.norm(P, axis=-1)
            dP = jnp.nan_to_num(dP)
            im = ax.pcolormesh(
                X,
                Y,
                dP,
                norm=LogNorm(vmin=1e-1, vmax=1e3),
                rasterized=True,
                zorder=-1,
            )
        else:
            PdB = 10.0 * jnp.log10(P / P0)
            im = ax.pcolormesh(X, Y, PdB, vmin=-50, vmax=5, rasterized=True, zorder=-1)

        cbar = fig.colorbar(im, ax=ax)

        if grad:
            cbar.ax.set_ylabel("Power gradient")
        else:
            cbar.ax.set_ylabel("Power (dB)")

        ax.set_ylabel("y coordinate")
        ax.set_title("With approximation" if approx else "Without approximation")

    axes[-1].set_xlabel("x coordinate")

    folder = Path(__file__).parent / "pgf"
    folder.mkdir(exist_ok=True)

    if grad:
        fig.savefig(folder / "power_gradient.pgf", dpi=300)
    else:
        fig.savefig(folder / "power_map.pgf", dpi=300)
