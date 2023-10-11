from copy import deepcopy as copy
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from matplotlib.colors import LogNorm
from utils import create_fig_for_paper

from differt2d.geometry import Point
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_obstacle()


def objective_function(received_power_per_receiver):
    acc = jnp.inf
    for p in received_power_per_receiver:
        p = p / P0  # Normalize power
        acc = jnp.minimum(acc, p)

    return acc


def loss(tx_coords, scene, *args, **kwargs):
    scene.emitters["Tx"].point = tx_coords
    return -objective_function(
        power for _, _, power in scene.accumulate_over_paths(*args, **kwargs)
    )


f_and_df = jax.value_and_grad(loss)

fig1, axes1 = create_fig_for_paper(
    2, 1, sharex=True, height_to_width_ratio=1.125, tight_layout=True
)
fig2, axes2 = create_fig_for_paper(
    1,
    4,
    sharex=True,
    sharey=True,
    tight_layout=True,
    columns=2,
    height_to_width_ratio=1.0 / 3.0,
)


annotate_kwargs = dict(color="black", fontsize=10, fontweight="bold", ha="center")
point_kwargs = dict(
    markersize=3, annotate_offset=(0, 0.05), annotate_kwargs=annotate_kwargs
)

scene.emitters = dict(
    Tx=Point(point=jnp.array([0.5, 0.7])),
)
scene.receivers = {
    r"Rx$_0$": Point(point=jnp.array([0.3, 0.1])),
    r"Rx$_1$": Point(point=jnp.array([0.5, 0.1])),
}

X, Y = scene.grid(n=600)

scenes = [scene, copy(scene)]  # Need a copy, because scenes will diverge

steps = 100  # In how many steps we hope to converge

alphas = jnp.logspace(0, 2, steps)  # Values between 1.0 and 100.0

tx_coords = jnp.array([0.5, 0.7])
optimizers = []
carries = []
for i, scene in enumerate(scenes):
    scene.emitters["Tx"].point = tx_coords
    optimizers.append(optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans()))
    carries.append((tx_coords, optimizers[i].init(tx_coords)))

for frame, alpha in enumerate(alphas):
    for i, approx in enumerate([False, True]):
        tx_coords, opt_state = carries[i]
        scenes[i].emitters["Tx"].point = tx_coords

        # Plotting prior to updating
        if frame % 20 == 0:
            if frame == 0:
                ax = axes1[i]
                ax.set_ylabel("y coordinate")
                if i == 1:
                    ax.set_xlabel("x coordinate")
                factor = 1.0
            elif approx:
                j = (frame // 20) - 1
                ax = axes2[j]
                ax.set_xlabel("x coordinate")
                if j == 0:
                    ax.set_ylabel("y coordinate")

                alpha_str = f"{alpha:.2e}"
                base, expo = alpha_str.split("e")
                expo = str(int(expo))  # Remove trailing zeros and +
                ax.set_title(f"Iterations: {frame:2d}")
                factor = 1.5
            else:
                continue  # We don't show further steps when no approx

            scenes[i].plot(
                ax,
                emitters_kwargs=point_kwargs,
                receivers=False,
                receivers_kwargs=dict(marker="x", **point_kwargs),
            )
            rx_kwargs = point_kwargs.copy()
            rx_kwargs.update(
                color="black",
                marker="x",
                annotate="Rx$_0$",
                annotate_offset=(-0.1 * factor, 0.0),
            )
            scenes[i].receivers["Rx$_0$"].plot(ax, **rx_kwargs)  # type: ignore[arg-type]
            rx_kwargs.update(annotate="Rx$_1$", annotate_offset=(+0.1 * factor, 0.0))
            scenes[i].receivers["Rx$_1$"].plot(ax, **rx_kwargs)  # type: ignore[arg-type]

            F = objective_function(
                power
                for _, power in scenes[i].accumulate_on_emitters_grid_over_paths(
                    X, Y, fun=received_power, max_order=0, approx=approx, alpha=alpha
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
            if frame == 0:
                cbar = fig1.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Objective function")
                ax.set_title(
                    "With approximation" if approx else "Without approximation"
                )

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

folder = Path(__file__).parent / "pgf"
folder.mkdir(exist_ok=True)

fig1.savefig(folder / "optimize_start.pgf", dpi=300, bbox_inches="tight")
fig2.savefig(folder / "optimize_steps.pgf", dpi=300, bbox_inches="tight")
