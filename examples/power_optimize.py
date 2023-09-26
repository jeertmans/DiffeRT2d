import jax.numpy as jnp
import matplotlib.pyplot as plt

from differt2d.geometry import Point
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_obstacle()

MIN = 0.01 * P0  # We want min -20 dB
MAX = 0.10 * P0  # We are not intersted in more that -10 dB

def thresholded_accumulate(*args, **kwargs):
    receivers = scene.receivers
    acc = 0.0
    for (r_key, receiver) in receivers.items():
        scene.receivers = {r_key: receiver}
        p = scene.accumulate_on_emitters_grid_over_paths(*args, **kwargs)
        p = jnp.where(p > MAX, MAX, p)
        p = jnp.where(p < MIN, 0.0, p)
        acc = acc + p

    scene.receivers = scene.receivers

    return p


fig, axes = plt.subplots(2, 1, sharex=True)

annotate_kwargs = dict(color="white", fontsize=12, fontweight="bold")

scene.receivers = dict(
    tx_0 = Point(point=jnp.array([0.3, 0.1])),
    tx_1 = Point(point=jnp.array([0.5, 0.1])),
)

X, Y = scene.grid(n=300)

for ax, approx in zip(axes, [False, True]):
    scene.plot(
        ax,
        emitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
        receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
    )

    P = thresholded_accumulate(
        X, Y, fun=received_power, approx=approx, alpha=10.0
    )

    PdB = 10.0 * jnp.log10(P / P0)

    im = ax.pcolormesh(X, Y, PdB, vmin=-50, vmax=5, zorder=-1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Power (dB)")

    ax.set_ylabel("y coordinate")
    ax.set_title("With approximation" if approx else "Without approximation")

axes[-1].set_xlabel("x coordinate")
plt.show()
