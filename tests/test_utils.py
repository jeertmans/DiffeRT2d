import chex
import jax.numpy as jnp

from differt2d.geometry import Path, Point
from differt2d.utils import received_power


def test_received_power():
    tx = rx = Point(xy=jnp.array([0.0, 0.0]))
    path = Path(
        xys=jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
    )

    expected = 0.3 / (2.0 * 2.0)
    got = received_power(tx, rx, path, [], r_coef=0.3, height=0.0)
    chex.assert_trees_all_close(got, expected)
