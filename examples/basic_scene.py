import matplotlib.pyplot as plt

from differt2d.scene import Scene


def main():
    ax = plt.gca()
    scene = Scene.basic_scene()
    scene = Scene.from_geojson(open("examples/example.geojson"), tx_loc="C", rx_loc="S")
    scene.plot(ax)

    for path in scene.all_paths():
        path.plot(ax)

    X, Y = scene.grid(n=150)

    Z = scene.accumulate_on_grid(X, Y)

    plt.pcolormesh(X, Y, Z)

    plt.show()


if __name__ == "__main__":
    main()
