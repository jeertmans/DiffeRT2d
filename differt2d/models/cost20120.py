"""Deep Learning model presented at the 8th COST ACTION CA20120 meeting in Helsinki."""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from ..geometry import normalize


class DeepSets(eqx.Module):
    """A MLP-based DeepSets model that returns a fixed sized vector from an arbitrary-sized sequence of fixed-sized objects."""

    object_size: int
    """The size of one object."""
    output_size: int
    """The size of the output vector."""
    phi: eqx.nn.Sequential
    """The MLP applied to each object in parallel."""
    rho: eqx.nn.MLP
    """The MLP applied to an permutation-invariant representation (sum of phi's) of all objects."""

    def __init__(
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

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, walls: Float[Array, "num_objects {self.object_size}"]
    ) -> Float[Array, "{self.output_size}"]:
        x = jax.vmap(self.phi)(walls)
        x = jnp.sum(x, axis=0)
        x = self.rho(x)

        return x


class LOL:
    pass


class PathGenerator(eqx.Module):
    """A recurrent model that returns a path of a given order."""

    order: int
    """The path order."""
    input_size: int
    """The input size."""
    cell: eqx.nn.LSTMCell
    """The recurrent unit to generate subsequent paths."""
    state_2_xy: eqx.nn.MLP
    """The layer(s) that convert a (cell) state into xy-coordinates."""

    def __init__(
        self,
        order: int,
        input_size: int,
        hidden_size: int = 10,
        width_size: int = 100,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jax.random.split(key, 2)

        self.order = order
        self.input_size = input_size
        self.cell = eqx.nn.GRUCell(
            input_size=input_size, hidden_size=hidden_size, key=key1
        )
        self.state_2_xy = eqx.nn.MLP(
            in_size=2 * hidden_size,
            out_size=2,
            width_size=width_size,
            depth=depth,
            key=key2,
        )

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        x: Float[Array, "{self.input_size}"],
        start: Float[Array, "2"],
        end: Float[Array, "2"],
    ) -> Float[Array, "{self.order}+2 2"]:
        @jax.jit
        @jaxtyped(typechecker=typechecker)
        def scan_fn(
            state: tuple[
                Float[Array, "{self.cell.hidden_size}"],
                Float[Array, "{self.cell.hidden_size}"],
            ],
            input_: Float[Array, "{self.input_size}"],
        ):
            state = self.cell(input_, state)
            xy = self.state_2_xy(jnp.vstack(state))
            return self.cell(input_, state), xy

        xs = jnp.tile(x, (self.order, 1))
        init_state = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )

        _, path = jax.lax.scan(scan_fn, init_state, xs)

        return jnp.vstack((start, path, end))


class PathEvaluator(eqx.Module):
    """A model that returns a probability that a path is valid."""

    order: int
    num_embeddings: int
    mlp: eqx.nn.MLP

    def __init__(
        self,
        order: int,
        num_embeddings: int,
        width_size: int = 500,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        self.order = order

        self.mlp = eqx.nn.MLP(
            in_size=2 + order * 2 + num_embeddings,
            out_size=1,
            width_size=width_size,
            depth=depth,
            final_activation=jax.nn.sigmoid,
            key=key,
        )

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        path: Float[Array, "{self.order}+2 2"],
        scene_embeddings: Float[Array, "{self.num_embeddings}"],
    ) -> Float[Array, " "]:
        x = jnp.concatenate((jnp.ravel(path), scene_embeddings))
        x = self.mlp(x)

        return x


class Model(eqx.Module):
    """Global Deep-Learning model."""

    # hyperparameters
    order: int
    num_embeddings: int

    # inference parameters
    num_paths: int
    threshold: float
    inference: bool

    # trainable
    walls_embed: WallsEmbed
    path_generator: PathGenerator
    path_evaluator: PathEvaluator
    # path_cell: eqx.nn.GRUCell

    def __init__(
        self,
        # Hyperparameters
        order: int = 1,
        num_embeddings: int = 100,
        # Inference parameters
        num_paths: int = 100,
        threshold: float = 0.1,
        inference: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.order = order
        self.num_paths = num_paths
        self.threshold = threshold
        self.inference = inference
        self.walls_embed = WallsEmbed(num_embeddings=num_embeddings, key=key1)
        self.path_generator = PathGenerator(
            order=order, input_size=num_embeddings, key=key2
        )
        self.path_evaluator = PathEvaluator(
            order=order, num_embeddings=num_embeddings, key=key3
        )
        # self.path_cell = eqx.nn.GRUCell(

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, x: Float[Array, "2+num_walls*2 2"]
    ) -> Float[Array, "num_paths {self.order}+2 2"]:
        # Processing input

        # [2]
        tx = x[0, :]
        rx = x[1, :]

        if self.order < 1:
            return jnp.vstack((tx, rx))

        # [num_walls, 2x2]
        walls = x[2:, :].reshape(-1, 4)
        starts = walls[:, 0, :]
        ends = walls[:, 1, :]

        # todo: pass those as parameters to force using specular reflection
        directions, _ = jax.vmap(normalize)(ends - starts)
        normals = directions.at[:, 0].set(directions[:, 1])
        normals = normals.at[:, 1].set(-directions[:, 0])

        walls_embeddings = self.walls_embed(walls)

        # Generate paths

        # Logic for one path:

        state = ...

        first_point_gen = ...

        other_points_gen = ...

        last_point_gen = ...

        while True:
            state = state_fun([state, tx, rx, walls_embeddings])

            path = []

            for _ in range(self.order):




        paths = []

        # paths = list_of_paths
        # probs = probab(paths)
        # paths = jnp.where(probs > threshold, paths, -1)

        def scan_fn(state: tuple[Float[Array, ""], ...], _: None = None):
            state = self.cell(walls_embeddings)
            xy = ...
            self.cell(input_, state), state

        if self.inference:
            paths = []
            probabilities = []

            while True:
                pass

            if len(paths) > 0:
                paths = jnp.stack(paths)
            else:
                paths = jnp.zeros((0, order + 2, 2))

            return paths, probibilities
        else:
            raise NotImplementedError
