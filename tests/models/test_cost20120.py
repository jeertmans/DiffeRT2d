import equinox as eqx
import jax
import pytest

from differt2d.models.cost20120 import Model


class TestModel:
    def test_random_input(self, key) -> None:
        order = 1
        num_paths = 3
        key_input, key_model = jax.random.split(key, 2)
        xy = jax.random.uniform(key_input, (10, 2))
        m = Model(order=order, num_paths=num_paths, key=key_model)

        _, paths = m(xy)

        assert paths.shape == (num_paths, order + 2, 2)

        m = eqx.nn.inference_mode(m)

        # TODO: test me
        # paths = m(xy)
        # assert paths.shape[1:] == (order + 2, 2)

    def test_no_walls(self, key) -> None:
        key_input, key_model = jax.random.split(key, 2)
        xy = jax.random.uniform(key_input, (2, 2))

        m = Model(order=0, num_paths=1, key=key_model)

        _, paths = m(xy)

        assert paths.shape == (1, 2, 2)

        m = eqx.nn.inference_mode(m)

        paths = m(xy)

        assert paths.shape == (1, 2, 2)

        m = Model(order=1, key=key_model)

        _, paths = m(xy)

        assert paths.shape == (0, 3, 2)

        m = eqx.nn.inference_mode(m)

        paths = m(xy)

        assert paths.shape == (0, 3, 2)

    def test_order(self, key) -> None:
        with pytest.raises(ValueError, match="Order must be greater or equal to 0"):
            _ = Model(order=-1, key=key)

    def test_num_embeddings(self, key) -> None:
        with pytest.raises(
            ValueError, match="Number of embeddings must be greater than 0"
        ):
            _ = Model(num_embeddings=0, key=key)

    def test_num_paths(self, key) -> None:
        with pytest.raises(ValueError, match="Number of paths must be greater than 0"):
            _ = Model(num_paths=-1, key=key)

    def test_threshold(self, key) -> None:
        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            _ = Model(threshold=1.5, key=key)

    def test_num_paths_warning(self, key) -> None:
        with pytest.warns(
            UserWarning, match="Consider setting 'num_paths = 1' when order is 0."
        ):
            _ = Model(order=0, num_paths=2, key=key)
