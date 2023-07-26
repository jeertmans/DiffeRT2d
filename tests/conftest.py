import jax
import pytest


@pytest.fixture
def seed():
    yield 1234


@pytest.fixture
def key(seed):
    yield jax.random.PRNGKey(seed)
