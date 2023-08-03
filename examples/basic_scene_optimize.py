import jax
import matplotlib.pyplot as plt

from differt2d.geometry import Point
from differt2d.optimize import minimize_many_random_uniform as minimize
from differt2d.scene import Scene, power


def main():
    ax = plt.gca()

    @jax.jit
    def loss_fun(x):
        _scene = Scene.basic_scene()
        _scene.rx = Point(point=x)
        total_power = _scene.accumulate_over_paths(function=power)
        return -total_power

    scene = Scene.basic_scene()

    key = jax.random.PRNGKey(1234)
    x, loss = minimize(loss_fun, key=key, n=2)

    scene.rx = Point(point=x)
    scene.plot(ax)

    for path in scene.all_paths():
        path.plot(ax)

    X, Y = scene.grid(n=150)

    Z = scene.accumulate_on_grid(X, Y)

    plt.pcolormesh(X, Y, Z)

    plt.show()


if __name__ == "__main__":
    main()
