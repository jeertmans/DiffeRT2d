from itertools import product

import jax
import optax
import pandas as pd

from differt2d.geometry import MinPath
from differt2d.scene import Scene


def min_path_tracing_loss(
    key: jax.random.PRNGKey, size: int, optimizer: optax.GradientTransformation
):
    key1, key2 = jax.random.split(key, 2)
    scene = Scene.random_uniform_scene(key1, size)
    return MinPath.from_tx_objects_rx(
        scene.tx, scene.objects, scene.rx, key2, optimizer=optimizer
    ).loss


def main():
    n = 1000
    seed = 1234
    key = jax.random.PRNGKey(seed)

    sizes = [1, 2, 3, 4, 5]
    optimizers = {
        "adam": optax.adam,
        "sgd": optax.sgd,
        "adagrad": optax.adagrad,
        "noisy_sgd": optax.noisy_sgd,
    }
    learning_rates = [1e-3, 1e-2, 1e-1, 1e-0]

    parameters = product(sizes, optimizers.keys(), learning_rates)

    results = {}

    for size, optimizer, learning_rate in parameters:
        print("size:", size)
        key, key_to_use = jax.random.split(key)

        opt = optimizers[optimizer](learning_rate)
        losses = jax.vmap(min_path_tracing_loss, in_axes=(0, None, None), out_axes=0)(
            jax.random.split(key_to_use, n), size, opt
        )
        results[(size, optimizer, learning_rate)] = losses

    index = pd.MultiIndex.from_tuples(
        results.keys(), names=["size", "optimizer", "learning_rate"]
    )
    df = pd.DataFrame(
        data=results.values(),
        index=index,
    )
    df.to_csv("optimizers.csv")


if __name__ == "__main__":
    main()
