import jax
import pytest
import matplotlib.pyplot as plt


@pytest.fixture
def ax():
    return plt.gca()


@pytest.fixture
def seed():
    yield 1234


@pytest.fixture
def key(seed):
    yield jax.random.PRNGKey(seed)
