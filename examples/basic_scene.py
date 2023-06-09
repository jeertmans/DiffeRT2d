# type: ignore
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from differt2d.geometry import Point, Wall
from differt2d.scene import Scene

jax.config.update("jax_enable_approx", False)

ax = plt.gca()

scene = Scene.basic_scene()

scene.plot(ax)

for path in scene.all_paths():
    path.plot(ax)

X, Y = scene.grid(n=10)
Z = scene.accumulate_on_grid(X, Y)

print(Z)

plt.pcolormesh(X, Y, Z)

# plt.axis("equal")

plt.show()
