import jax
import jax.numpy as jnp
import numpy as np
import optax

from tqdm import trange, tqdm

from differt2d.geometry import Point
from differt2d.scene import Scene
from differt2d.utils import P0, received_power

scene = Scene.basic_scene()


def objective_function(received_power_per_receiver):
    acc = jnp.inf
    for p in received_power_per_receiver:
        p = p / P0  # Normalize power
        acc = jnp.minimum(acc, p)

    return acc


def loss(tx_coords, scene, *args, **kwargs):
    scene.emitters["tx"].point = tx_coords
    return -objective_function(
        power for _, _, power in scene.accumulate_over_paths(*args, **kwargs)
    )


f_and_df = jax.value_and_grad(loss)

scene.emitters = dict(
    tx=Point(point=jnp.array([0.5, 0.7])),
)

X, Y = scene.grid(n=100)

steps = 100  # In how many steps we hope to converge

n_receivers = 3

alphas = jnp.logspace(0, 2, steps)  # Values between 1.0 and 100.0

stats = {}

for n_receivers in tqdm([2, 3, 4]):
    stats[n_receivers] = {True: [], False: []}
    for repeat in trange(50, leave=False):
        scene.receivers = {
            f"rx_{i}": Point(point=jnp.asarray(np.random.rand(2)))
            for i in range(n_receivers)
        }
        for starting_point in range(50):
            tx_coords = np.random.rand(2)
            scene.emitters["tx"].point = tx_coords
            F = objective_function(
                power
                for _, power in scene.accumulate_on_emitters_grid_over_paths(
                    X, Y, fun=received_power, max_order=0, approx=False,
                )
            )

            indices = jnp.unravel_index(F.argmax(), F.shape)
            expected = jnp.array([X[indices], Y[indices]])
            #print("expected solution:", expected)

            for approx in [False, True]:
                opt = optax.chain(optax.adam(learning_rate=0.01), optax.zero_nans())
                opt_state = opt.init(tx_coords)
                for i, alpha in enumerate(alphas):
                    scene.emitters["tx"].point = tx_coords
                    loss, grads = f_and_df(
                        tx_coords,
                        scene,
                        fun=received_power,
                        max_order=0,
                        approx=approx,
                        alpha=alpha,
                    )
                    updates, opt_state = opt.update(grads, opt_state)
                    tx_coords = tx_coords + updates

                #print("after convergence for approx", approx, "found:", tx_coords)
                converged = jnp.linalg.norm(expected - tx_coords) < 1e-1
                stats[n_receivers][approx].append(bool(converged))
                #print(converged)


import json

with open("results_basic.json", "w") as f:
    json.dump(stats, f)
