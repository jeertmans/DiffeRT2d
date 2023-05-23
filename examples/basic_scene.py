import jax.numpy as jnp
import matplotlib.pyplot as plt

from differt2d.geometry import Point, Wall
from differt2d.scene import Scene

tx = Point(point=jnp.array([0.1, 0.1]))
rx = Point(point=jnp.array([0.5, 0.5]))

walls = [
    # Outer walls
    Wall(points=jnp.array([[0.0, 0.0], [1.0, 0.0]])),
    Wall(points=jnp.array([[1.0, 0.0], [1.0, 1.0]])),
    Wall(points=jnp.array([[1.0, 1.0], [0.0, 1.0]])),
    Wall(points=jnp.array([[0.0, 1.0], [0.0, 0.0]])),
    # Small room
    Wall(points=jnp.array([[0.4, 0.0], [0.4, 0.4]])),
    Wall(points=jnp.array([[0.4, 0.4], [0.3, 0.4]])),
    Wall(points=jnp.array([[0.1, 0.4], [0.0, 0.4]])),
]

ax = plt.gca()

scene = Scene(tx=tx, rx=rx, objects=walls)

scene.plot(ax)

for path in scene.all_paths():
    path.plot(ax)

plt.axis("equal")

plt.show()
