from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float

from differt2d.geometry import RIS, MinPath
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene()
ris = RIS(
    xys=jnp.array([[0.5, 0.3], [0.5, 0.7]]),
    phi=jnp.pi / 4,  # type: ignore[reportArgumentType]
)
scene = scene.add_objects(ris)

fig, ax = plt.subplots()

annotate_kwargs = dict(color="white", fontsize=12, fontweight="bold")

key = jax.random.PRNGKey(1234)
X, Y = scene.grid(n=300)

scene.plot(
    ax,
    transmitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
    receivers=False,
)

P: Float[Array, "300 300"] = scene.accumulate_on_receivers_grid_over_paths(
    X,
    Y,
    fun=received_power,
    path_cls=MinPath,
    order=1,
    reduce_all=True,
    path_cls_kwargs={"steps": 1000},
    key=key,
)  # type: ignore

PdB = 10.0 * jnp.log10(P / P0)

im = ax.pcolormesh(
    X,
    Y,
    PdB,
    vmin=-50,
    vmax=5,
    zorder=-1,
    rasterized=True,
    antialiased=True,
)
cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Power (dB)")

ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")

folder = Path(__file__).parent
static = folder / "static"
static.mkdir(exist_ok=True)

fig.savefig(folder / "ris_power_map.pdf", dpi=300, bbox_inches="tight")
fig.savefig(static / "ris_power_map.png", dpi=300, bbox_inches="tight")
