from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float
from matplotlib.colors import LogNorm

from differt2d.geometry import Point
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_obstacle()


def objective_function(
    received_power_per_receiver: Iterator[Float[Array, " *batch"]],
) -> Float[Array, " *batch"]:
    acc = jnp.array(jnp.inf)
    for p in received_power_per_receiver:
        p = p / P0  # Normalize power
        acc = jnp.minimum(acc, p)

    return acc


def loss(
    tx_coords: Float[Array, "2"], scene: Scene, *args: Any, **kwargs: Any
) -> Float[Array, " "]:
    scene = scene.with_transmitters(tx=Point(xy=tx_coords))
    return -objective_function(
        power for _, _, power in scene.accumulate_over_paths(*args, **kwargs)
    )


f_and_df = jax.value_and_grad(loss)

fig, axes = plt.subplots(
    1,
    4,
    figsize=(6.4, 2.4),
    sharex=True,
    sharey=True,
    tight_layout=True,
)


annotate_kwargs = dict(color="black", fontsize=10, fontweight="bold", ha="center")
point_kwargs = dict(
    markersize=3, annotate_offset=(0, 0.05), annotate_kwargs=annotate_kwargs
)

scene = scene.with_receivers(
    rx_0=Point(xy=jnp.array([0.3, 0.1])), rx_1=Point(xy=jnp.array([0.5, 0.1]))
)

X, Y = scene.grid(n=600)

steps = 100  # In how many steps we hope to converge

alphas = jnp.logspace(0, 2, steps)  # Values between 1.0 and 100.0

tx_coords = jnp.array([0.5, 0.7])
optim = optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans())
opt_state = optim.init(tx_coords)

for frame, alpha in enumerate(alphas):
    scene = scene.with_transmitters(tx=Point(xy=tx_coords))
    # Plotting prior to updating
    if frame % 20 == 0 and frame > 0:
        j = (frame // 20) - 1
        ax = axes[j]
        ax.set_xlabel("x coordinate")
        if j == 0:
            ax.set_ylabel("y coordinate")

        alpha_str = f"{alpha:.2e}"
        base, expo = alpha_str.split("e")
        expo = str(int(expo))  # Remove trailing zeros and +
        ax.set_title(f"Iterations: {frame:2d}")
        factor = 1.5

        scene.plot(
            ax,
            transmitters_kwargs=point_kwargs,
            receivers=False,
            receivers_kwargs=dict(marker="x", **point_kwargs),
        )
        rx_kwargs = point_kwargs.copy()
        rx_kwargs.update(
            color="black",  # type: ignore[arg-type]
            marker="x",  # type: ignore[arg-type]
            annotate="rx_0",  # type: ignore[arg-type]
            annotate_offset=(-0.1 * factor, 0.0),  # type: ignore[arg-type]
        )
        scene.receivers["rx_0"].plot(ax, **rx_kwargs)  # type: ignore[arg-type]
        rx_kwargs.update(annotate="rx_1", annotate_offset=(+0.1 * factor, 0.0))  # type: ignore[arg-type]
        scene.receivers["rx_1"].plot(ax, **rx_kwargs)  # type: ignore[arg-type]

        F = objective_function(
            power  # type: ignore
            for _, power in scene.accumulate_on_transmitters_grid_over_paths(
                X, Y, fun=received_power, max_order=0, approx=True, alpha=alpha
            )
        )

        im = ax.pcolormesh(
            X,
            Y,
            F,
            norm=LogNorm(vmin=1e-4, vmax=1e0),
            zorder=-1,
            rasterized=True,
            antialiased=True,
        )

    loss, grads = f_and_df(
        tx_coords,
        scene,
        fun=received_power,
        max_order=0,
        approx=True,
        alpha=alpha,
    )
    updates, opt_state = optim.update(grads, opt_state)
    tx_coords = tx_coords + updates

folder = Path(__file__).parent
static = folder / "static"
static.mkdir(exist_ok=True)

fig.savefig(folder / "optimize_steps.pdf", dpi=300, bbox_inches="tight")
fig.savefig(static / "optimize_steps.png", dpi=300, bbox_inches="tight")
