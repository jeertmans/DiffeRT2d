import jax.numpy as jnp
import matplotlib.pyplot as plt

from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.square_scene_with_wall()

fig, axes = plt.subplots(2, 1, sharex=True)

annotate_kwargs = dict(fontsize=12, fontweight="bold")

x = jnp.linspace(0.2, 0.8, 200)
y = jnp.array([0.5])

X, Y = jnp.meshgrid(x, y)

scene.plot(
    axes[0],
    emitters_kwargs=dict(annotate_kwargs=annotate_kwargs),
    receivers_kwargs=dict(annotate_kwargs=annotate_kwargs),
)

P = scene.accumulate_on_receivers_grid_over_paths(
    X, Y, fun=received_power, approx=False, min_order=0, max_order=0
)

PdB = 10.0 * jnp.log10(P.reshape(-1) / P0)

axes[1].plot(x, PdB, label="Without")

for alpha in jnp.logspace(-1, 3, 10):
    P = scene.accumulate_on_receivers_grid_over_paths(
        X, Y, fun=received_power, approx=True, alpha=alpha, min_order=0, max_order=0
    )

    PdB = 10.0 * jnp.log10(P.reshape(-1) / P0)

    axes[1].plot(x, PdB, label="With + alpha = " + str(alpha))

axes[1].set_ylabel("Power (dB)")
axes[1].set_title("With appro")

axes[-1].set_xlabel("x coordinate")
plt.legend()
plt.show()
