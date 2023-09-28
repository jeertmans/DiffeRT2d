from copy import deepcopy as copy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matplotlib.animation import FuncAnimation

from differt2d.geometry import Point
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_obstacle()


def objective_function(received_power_per_receiver):
    """Objective function, that wants to maximise the sum of receiver power by each user
    (in dB)."""
    acc = 0.0
    for p in received_power_per_receiver:
        p = p / P0  # Normalize power
        acc = acc + jnp.log10(p)

    return 10.0 * acc


def loss(tx_coords, scene, *args, **kwargs):
    """Loss function, to be minimized."""
    scene.emitters["tx"].point = tx_coords
    return -objective_function(
        power for _, _, power in scene.accumulate_over_paths(*args, **kwargs)
    )


f_and_df = jax.value_and_grad(loss)

fig, axes = plt.subplots(2, 1, sharex=True)

annotate_kwargs = dict(color="red", fontsize=12, fontweight="bold")

scene.emitters = dict(
    tx=Point(point=jnp.array([0.5, 0.7])),
)
scene.receivers = dict(
    tx_0=Point(point=jnp.array([0.3, 0.1])),
    tx_1=Point(point=jnp.array([0.5, 0.1])),
)

X, Y = scene.grid(n=300)

im_artists = []
emitter_artists = []
annotate_artists = []
scenes = [scene, copy(scene)]  # Need a copy, because scenes will diverge

for ax, approx, scene in zip(axes, [False, True], scenes):
    scene_artists = scene.plot(
        ax,
        emitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
        receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
    )
    emitter_artists.append(scene_artists[0])
    annotate_artists.append(scene_artists[1])

    im = ax.pcolormesh(X, Y, jnp.zeros_like(X), vmin=-60, vmax=5, zorder=-1)
    im_artists.append(im)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Objective function")
    ax.set_ylabel("y coordinate")
    ax.set_title("With approximation" if approx else "Without approximation")

axes[-1].set_xlabel("x coordinate")

steps = 100

alphas = jnp.logspace(0, 2, steps)

optimizers = [
    optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans()) for _ in scenes
]
carries = [
    (scene.emitters["tx"].point, opt.init(scene.emitters["tx"].point))
    for opt, scene in zip(optimizers, scenes)
]


def func(alpha):
    for i, approx in enumerate([False, True]):
        tx_coords, opt_state = carries[i]
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
        scenes[i].emitters["tx"].point = tx_coords
        emitter_artists[i].set_data([tx_coords[0]], [tx_coords[1]])
        annotate_artists[i].set_x(tx_coords[0])
        annotate_artists[i].set_y(tx_coords[1])

        F = objective_function(
            power
            for _, power in scenes[i].accumulate_on_emitters_grid_over_paths(
                X, Y, fun=received_power, max_order=0, approx=approx, alpha=alpha
            )
        )
        im_artists[i].set_array(F)

        if approx:
            axes[i].set_title(f"With approximation - $\\alpha={alpha:.2e}$")


anim = FuncAnimation(fig, func=func, frames=alphas)
# anim.save("anim.gif")
plt.show()
