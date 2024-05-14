from collections.abc import Iterator

import jax
import matplotlib.pyplot as plt
import pytest
from jaxtyping import PRNGKeyArray
from matplotlib.axes import Axes


@pytest.fixture
def ax() -> Iterator[Axes]:
    yield plt.gca()


@pytest.fixture
def seed() -> Iterator[int]:
    yield 1234


@pytest.fixture
def key(seed: int) -> Iterator[PRNGKeyArray]:
    yield jax.random.PRNGKey(seed)
