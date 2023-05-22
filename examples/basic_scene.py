import jax.numpy as jnp
import matplotlib.pyplot as plt

from differt2d.geometry import Point, Wall
from differt2d.scene import Scene

tx = Point(point=jnp.array([0.1, 0.1]))
rx = Point(point=jnp.array([0.7, 0.7]))

walls = [
    # Outer walls
    Wall(points=jnp.array([[0., 0.], [1., 0.]])),
    Wall(points=jnp.array([[1., 0.], [1., 1.]])),
    Wall(points=jnp.array([[1., 1.], [0., 1.]])),
    Wall(points=jnp.array([[0., 1.], [0., 0.]])),
    # Small room
    Wall(points=jnp.array([[.4, 0.], [.4, .4]])),
    Wall(points=jnp.array([[.4, .4], [.3, .4]])),
    Wall(points=jnp.array([[.1, .4], [0., .4]])),
]

ax = plt.gca()

scene = Scene(tx=tx, rx=rx, objects=walls)

scene.plot(ax)

for path in scene.all_paths():
    path.plot(ax)

plt.show()

