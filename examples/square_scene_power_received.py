import matplotlib.pyplot as plt
import typer

from differt2d.scene import Scene


def main(resolution: int = 150):
    ax = plt.gca()
    scene = Scene.square_scene()
    scene.plot(ax)

    for path in scene.all_paths():
        path.plot(ax)

    X, Y = scene.grid(n=resolution)

    Z = scene.accumulate_on_grid(X, Y, min_order=1, max_order=1)

    plt.pcolormesh(X, Y, Z)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
