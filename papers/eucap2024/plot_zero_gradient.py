from pathlib import Path

import jax.numpy as jnp
from jaxtyping import Array, Float

from differt2d.scene import Scene
from differt2d.utils import P0, received_power
from utils import create_fig_for_paper  # type: ignore[reportMissingImports]

scene = Scene.square_scene_with_wall()
scene = scene.with_transmitters(Tx=scene.transmitters["tx"])

fig, ax = create_fig_for_paper(1, 1, height_to_width_ratio=0.6, tight_layout=True)

annotate_kwargs = dict(color="black", fontsize=10, fontweight="bold", ha="center")
point_kwargs = dict(
    markersize=3, annotate_offset=(0, 0.05), annotate_kwargs=annotate_kwargs
)

X, Y = scene.grid(n=600)

scene.plot(
    ax,
    transmitters_kwargs=point_kwargs,
    receivers=False,
)

P: Float[Array, "600 600"] = scene.accumulate_on_receivers_grid_over_paths(
    X, Y, fun=received_power, max_order=0, reduce_all=True, approx=False
)  # type: ignore

PdB = 10.0 * jnp.log10(P / P0)

im = ax.pcolormesh(
    X, Y, PdB, vmin=-50, vmax=5, rasterized=True, antialiased=True, zorder=-1
)
cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Power (dB)")
ax.annotate(r"$\nabla = 0$", (0.6, 0.5))
ax.annotate(
    "",
    xy=(0.5, 0.05),
    xycoords="data",
    xytext=(0.7, 0.25),
    textcoords="data",
    arrowprops=dict(arrowstyle="<->", connectionstyle="angle3,angleA=90,angleB=0"),
)
ax.annotate(
    "",
    xy=(0.5, 0.95),
    xycoords="data",
    xytext=(0.7, 0.75),
    textcoords="data",
    arrowprops=dict(arrowstyle="<->", connectionstyle="angle3,angleA=-90,angleB=0"),
)

ax.set_ylabel("y coordinate")
ax.set_xlabel("x coordinate")

folder = Path(__file__).parent / "pgf"
folder.mkdir(exist_ok=True)

fig.savefig(folder / "zero_gradient.pgf", dpi=300, bbox_inches="tight")
