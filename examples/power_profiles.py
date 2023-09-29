import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array

from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_wall()

fig, axes = plt.subplots(2, 1, sharex=True)

# Scene and power map

annotate_kwargs = dict(color="red", fontsize=12, fontweight="bold")

scene.plot(
    axes[0],
    emitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
    receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
)

X, Y = scene.grid(n=300)
P: Array = scene.accumulate_on_receivers_grid_over_paths(
    X,
    Y,
    fun=received_power,
    reduce=True,
    approx=False,
    min_order=0,
    max_order=0,
)

PdB = 10.0 * jnp.log10(P / P0)

axes[0].pcolormesh(X, Y, PdB, vmin=-50, vmax=5, zorder=-1)
axes[0].set_ylabel("y coordinate")
axes[1].set_title("Without approx.")

# Profiles

x = jnp.linspace(0.2, 0.8, 200)
y = jnp.array([0.5])

X, Y = jnp.meshgrid(x, y)

P = scene.accumulate_on_receivers_grid_over_paths(
    X, Y, fun=received_power, reduce=True, approx=False, min_order=0, max_order=0
)

PdB = 10.0 * jnp.log10(P.reshape(-1) / P0)

axes[1].plot(x, PdB, label="Without")

for alpha in [1.0, 10.0, 100.0, 1000.0]:
    P = scene.accumulate_on_receivers_grid_over_paths(
        X,
        Y,
        fun=received_power,
        reduce=True,
        approx=True,
        alpha=alpha,
        min_order=0,
        max_order=0,
    )

    PdB = 10.0 * jnp.log10(P.reshape(-1) / P0)

    axes[1].plot(x, PdB, label=f"With + $\\alpha = {alpha:.0e}$")

axes[1].set_ylabel("Power (dB)")
axes[1].set_title("With approx.")
axes[1].set_ylim([-20, 0])

axes[-1].set_xlabel("x coordinate")
plt.legend()
plt.show()
