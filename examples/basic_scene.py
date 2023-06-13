import matplotlib.pyplot as plt

from differt2d.scene import Scene


def main():
    ax = plt.gca()
    scene = Scene.basic_scene()
    scene.plot(ax)

    for path in scene.all_paths():
        path.plot(ax)

    X, Y = scene.grid(n=100)

    Z = scene.accumulate_on_grid(X, Y)

    plt.pcolormesh(X, Y, Z)

    plt.show()


if __name__ == "__main__":
    main()
