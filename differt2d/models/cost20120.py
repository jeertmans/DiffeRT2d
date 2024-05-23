"""
Deep Learning model presented at the 8th COST ACTION CA20120 meeting in Helsinki.

The model takes as input a scene made of (tx, rx, walls...) in 2D
and returns a sequence of paths that link tx and rx, while
undergoing a fixed number (= model.order) of reflections.
"""

import warnings
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from ..geometry import normalize


class DeepSets(eqx.Module):
    """
    A MLP-based DeepSets model that returns a fixed sized vector from an arbitrary-sized sequence of fixed-sized objects.

    :param object_size: The size of one object.
    :param output_size: The size of the output vector.
    :param phi_width_size: The width size of the phi (1st) MLP.
    :param phi_depth: The depth of the phi (1st) MLP.
    :param intermediate_size: The size of the vector between the two MLPs.
    :param rho_width_size: The width size of the rho (2nd) MLP.
    :param rho_depth: The depth of the rho (2nd) MLP.
    :param key: The random key to be used.
    """

    object_size: int = eqx.field(static=True)
    """The size of one object."""
    output_size: int = eqx.field(static=True)
    """The size of the output vector."""
    phi: eqx.nn.MLP
    """The MLP applied to each object in parallel."""
    rho: eqx.nn.MLP
    """The MLP applied to an permutation-invariant representation (sum of phi's) of all objects."""

    def __init__(  # noqa: D107
        self,
        object_size: int,
        output_size: int,
        phi_width_size: int = 500,
        phi_depth: int = 3,
        intermediate_size: int = 500,
        rho_width_size: int = 500,
        rho_depth: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jax.random.split(key, 2)
        self.object_size = object_size
        self.output_size = output_size
        self.phi = eqx.nn.MLP(
            in_size=object_size,
            out_size=intermediate_size,
            width_size=phi_width_size,
            depth=phi_depth,
            key=key1,
        )
        self.rho = eqx.nn.MLP(
            in_size=intermediate_size,
            out_size=output_size,
            width_size=rho_width_size,
            depth=rho_depth,
            key=key2,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(  # noqa: D102
        self, walls: Float[Array, "num_objects {self.object_size}"]
    ) -> Float[Array, " {self.output_size}"]:
        x = jax.vmap(self.phi)(walls)
        x = jnp.sum(x, axis=0)
        x = self.rho(x)

        return x


class PathGenerator(eqx.Module):
    """
    A recurrent model that returns a path of a given order and a probability that this path is valid.

    :param order: The order of the path.
    :param num_embeddings: The number of embeddings to represent the scene.
    :param hidden_size: The size of the hidden layers in the cells.
    :param width_size: The width size of the MLPs.
    :param depth: The depth of the MLPs.
    :param key: The random key to be used.
    """

    order: int = eqx.field(static=True)
    """The path order."""
    num_embeddings: int = eqx.field(static=True)
    """Number of embedding points to represent the scene (used for path validation)."""
    hidden_size: int = eqx.field(static=True)
    """The hidden size of the LSTM cell."""
    cell: eqx.nn.LSTMCell
    """The recurrent unit to generate subsequent paths."""
    state_2_xy: eqx.nn.MLP
    """The layer(s) that convert a (cell) state into xy-coordinates."""
    state_2_probability: eqx.nn.MLP
    """The layer(s) that convert a (cell) state into probability that a path is valid."""

    def __init__(  # noqa: D107
        self,
        order: int,
        num_embeddings: int,
        hidden_size: int,
        width_size: int = 3,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jax.random.split(key, 2)

        self.order = order
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(
            input_size=2,  # The newly generated xy-coordinates
            hidden_size=hidden_size,
            key=key1,
        )
        self.state_2_xy = eqx.nn.MLP(
            in_size=2 * hidden_size,  # Input is hidden state from cell
            out_size=2,  # Output is xy-coordinates
            width_size=width_size,
            depth=depth,
            key=key2,
        )
        self.state_2_probability = eqx.nn.MLP(
            in_size=2 * hidden_size + num_embeddings,
            out_size="scalar",  # Output is probability that path is valid
            width_size=width_size,
            depth=depth,
            final_activation=jax.nn.sigmoid,  # Output must be between 0 and 1
            key=key2,
        )

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(  # noqa: D102
        self,
        init_state: tuple[
            Float[Array, " {self.hidden_size}"], Float[Array, " {self.hidden_size}"]
        ],
        tx: Float[Array, "2"],
        rx: Float[Array, "2"],
        scene: Float[Array, "num_walls 8"],
        scene_embeddings: Float[Array, " {self.num_embeddings}"],
    ) -> tuple[Float[Array, " "], Float[Array, "{self.order}+2 2"]]:
        # Generate new state with path starting at TX
        init_state = self.cell(tx, init_state)

        @jax.jit
        @jaxtyped(typechecker=typechecker)
        def scan_fn(
            state: tuple[
                Float[Array, " hidden_size"],
                Float[Array, " hidden_size"],
            ],
            _: None,
        ):
            # TODO: use scene.normals and others to force specular reflection
            xy = self.state_2_xy(jnp.concatenate(state))
            state = self.cell(xy, state)
            return state, xy

        final_state, path = jax.lax.scan(scan_fn, init_state, length=self.order)

        # Generate final state with path ending at RX
        final_state = self.cell(rx, final_state)

        p = self.state_2_probability(jnp.concatenate((*final_state, scene_embeddings)))

        return p, jnp.vstack((tx, path, rx))


class Model(eqx.Module):
    """
    Global Deep-Learning model.

    :param order: The order of the path(s).
    :param num_embeddings: The number of embeddings to represent the scene.
    :param hidden_size: The size of the hidden layers.
    :param num_paths: The number of paths to return during training.
    :param inference: The threshold to use on paths during inference.
    :param inference: Whether this model should be used for inference.
    :param key: The random key to be used.
    """

    # Hyperparameters
    order: int = eqx.field(static=True)
    """The order of the path(s)."""
    num_embeddings: int = eqx.field(static=True)
    """ The number of embeddings to represent the scene."""

    # Training parameters
    num_paths: int = eqx.field(static=True)
    """The number of paths to return during training."""

    # Inference parameters
    threshold: float = eqx.field(static=True)
    """The threshold to use on paths during inference."""
    inference: bool = eqx.field(static=True)
    """Whether this model should be used for inference."""

    # Trainable
    scene_embed: DeepSets
    """Layer(s) that extract information about the scene."""
    path_generator: PathGenerator
    """Layer(s) that generate one path orde a specific order."""
    embeddings_2_state: eqx.nn.MLP
    """Layer(s) that map embedding to one of the cell states."""
    cell: eqx.nn.LSTMCell
    """Layer(s) that provide some memory over generated paths."""

    def __init__(  # noqa: D107
        self,
        # Hyperparameters
        order: int = 1,
        num_embeddings: int = 100,
        hidden_size: int = 100,
        # Training parameters
        num_paths: int = 5,
        # Inference parameters
        threshold: float = 0.5,
        inference: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.order = order
        self.num_embeddings = num_embeddings
        self.num_paths = num_paths
        self.threshold = threshold
        self.inference = inference
        self.scene_embed = DeepSets(
            object_size=2 + 2 + 2 + 2,  # start, end, normal, direction
            output_size=num_embeddings,
            key=key1,
        )
        self.path_generator = PathGenerator(
            order=order,
            num_embeddings=num_embeddings,
            hidden_size=hidden_size,
            key=key2,
        )
        self.embeddings_2_state = eqx.nn.MLP(
            in_size=num_embeddings,
            out_size=2 * hidden_size,
            width_size=200,
            depth=3,
            key=key3,
        )
        self.cell = eqx.nn.LSTMCell(
            input_size=(2 + order) * 2, hidden_size=hidden_size, key=key4
        )

    def __check_init__(self):  # noqa: D105
        if self.order < 0:
            raise ValueError(f"Order must be greater or equal to 0, got {self.order}.")
        if self.num_embeddings <= 0:
            raise ValueError(
                f"Number of embeddings must be greater than 0, got {self.num_embeddings}."
            )
        if self.num_paths <= 0:
            raise ValueError(
                f"Number of paths must be greater than 0, got {self.num_paths}."
            )
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError(
                f"Threshold must be between 0 and 1, got {self.threshold}."
            )
        if self.order == 0 and self.num_paths > 1:
            warnings.warn(
                "Consider setting 'num_paths = 1' when order is 0.",
                UserWarning,
                stacklevel=2,
            )

    @eqx.filter_jit
    @jaxtyped(typechecker=None)  # TODO: fixme
    def __call__(  # noqa: D102
        self, xy: Float[Array, "two_plus_num_walls_times_two 2"]
    ) -> Union[
        tuple[
            Float[Array, " {self.num_paths}"],
            Float[Array, "{self.num_paths} {self.order}+2 2"],
        ],
        Float[Array, "num_paths {self.order}+2 2"],
    ]:
        assert xy.shape[0] >= 2, "Scene must at least have two points: tx and rx."

        # -- Step 1: Processing input

        # [2]
        tx = xy[0, :]
        rx = xy[1, :]

        # [num_walls, 2x2]
        walls = xy[2:, :].reshape(-1, 4)

        # Handle empty scene
        if walls.size == 0:
            if self.order == 0:
                paths = jnp.vstack((tx, rx))
                probabilities = jnp.ones((1,))
            else:
                paths = jnp.empty((0, self.order + 2, 2))
                probabilities = jnp.empty((0,))

            if self.inference:
                return paths
            else:
                return probabilities, paths

        # Let's extract walls' normal and direction vectors, both normalized
        starts = walls[:, 0:2]
        ends = walls[:, 2:4]
        directions, _ = jax.vmap(normalize)(ends - starts)
        normals = directions.at[:, 0].set(directions[:, 1])
        normals = normals.at[:, 1].set(-directions[:, 0])

        # [num_walls, 4x2]
        scene = jnp.hstack((walls, normals, directions))

        # [self.num_embeddings]
        scene_embeddings = self.scene_embed(scene)

        # -- Step 2: Generating paths

        @jax.jit
        @jaxtyped(typechecker=typechecker)
        def scan_fn(
            state: tuple[
                Float[Array, " hidden_size"],
                Float[Array, " hidden_size"],
            ],
            _: None,
        ) -> tuple[
            tuple[
                Float[Array, " hidden_size"],
                Float[Array, " hidden_size"],
            ],
            tuple[Float[Array, " "], Float[Array, "order_plus_2 2"]],
        ]:
            p, path = self.path_generator(state, tx, rx, scene, scene_embeddings)
            state = self.cell(jnp.ravel(path), state)
            return state, (p, path)

        hidden_state, cell_state = jnp.split(
            self.embeddings_2_state(scene_embeddings), 2
        )

        init_state = (
            hidden_state,
            cell_state,
        )

        if self.inference:
            paths = jnp.zeros((0, self.order + 2, 2))
            state = init_state

            while True:
                state, (p, path) = scan_fn(state, None)

                if p >= self.threshold:
                    paths = jnp.vstack((paths, path[None, ...]))
                else:
                    break

            return paths
        else:
            _, (probabilities, paths) = jax.lax.scan(
                scan_fn, init_state, length=self.num_paths
            )

            return probabilities, paths
