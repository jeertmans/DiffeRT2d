from copy import deepcopy as copy
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from utils import setup_fig_for_paper

from differt2d.geometry import Point
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_obstacle()


def objective_function(received_power_per_receiver):
    acc = 0.0
    for p in received_power_per_receiver:
        p = p / P0  # Normalize power
        acc = acc + jnp.log10(p)

    return 10.0 * acc


def loss(tx_coords, scene, *args, **kwargs):
    scene.emitters["tx"].point = tx_coords
    return -objective_function(
        power for _, _, power in scene.accumulate_over_paths(*args, **kwargs)
    )


f_and_df = jax.value_and_grad(loss)

fig1, axes1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
fig2, axes2 = plt.subplots(1, 4, sharex=True, sharey=True, tight_layout=True)
setup_fig_for_paper(fig1)
setup_fig_for_paper(fig2, columns=2, height_to_width_ratio=1.0 / 3.0)

annotate_kwargs = dict(color="red", fontsize=12, fontweight="bold")

scene.emitters = dict(
    tx=Point(point=jnp.array([0.5, 0.7])),
)
scene.receivers = {
    r"rx$_0$": Point(point=jnp.array([0.3, 0.1])),
    r"rx$_1$": Point(point=jnp.array([0.5, 0.1])),
}

X, Y = scene.grid(n=600)

scenes = [scene, copy(scene)]  # Need a copy, because scenes will diverge

steps = 100  # In how many steps we hope to converge

alphas = jnp.logspace(0, 2, steps)  # Values between 1.0 and 100.0

tx_coords = jnp.array([0.5, 0.7])
optimizers = []
carries = []
for i, scene in enumerate(scenes):
    scene.emitters["tx"].point = tx_coords
    optimizers.append(optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans()))
    carries.append((tx_coords, optimizers[i].init(tx_coords)))

for frame, alpha in enumerate(alphas):
    for i, approx in enumerate([False, True]):
        tx_coords, opt_state = carries[i]
        scenes[i].emitters["tx"].point = tx_coords

        # Plotting prior to updating
        if frame % 20 == 0:
            if frame == 0:
                ax = axes1[i]
                ax.set_ylabel("y coordinate")
                if i == 1:
                    ax.set_xlabel("x coordinate")
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
            else:
                continue  # We don't show further steps when no approx

            scenes[i].plot(
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

            im = ax.pcolormesh(X, Y, F, vmin=-60, vmax=5, zorder=-1, rasterized=True)
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

fig1.savefig(folder / "optimize_start.pgf", dpi=300)
fig2.savefig(folder / "optimize_steps.pgf", dpi=300)
