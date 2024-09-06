import jax
import matplotlib.pyplot as plt
import pytest
from jaxtyping import PRNGKeyArray
from matplotlib.axes import Axes


@pytest.fixture
def ax() -> Axes:
    return plt.gca()


@pytest.fixture(scope="session")
def seed() -> int:
    return 1234


@pytest.fixture(scope="session")
def key(seed: int) -> PRNGKeyArray:
    return jax.random.PRNGKey(seed)
