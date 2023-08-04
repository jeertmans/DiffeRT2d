import matplotlib.pyplot as plt
import typer

from differt2d.scene import Scene


def main(min_order: int = 0, max_order: int = 1, resolution: int = 150):
    ax = plt.gca()
    scene = Scene.basic_scene()
    scene.plot(ax)

    for path in scene.all_paths():
        path.plot(ax)

    X, Y = scene.grid(n=resolution)

    Z = scene.accumulate_on_grid(X, Y, min_order=min_order, max_order=max_order)

    plt.pcolormesh(X, Y, Z)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
