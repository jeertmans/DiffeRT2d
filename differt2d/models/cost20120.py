"""Deep Learning model presented at the 8th COST ACTION CA20120 meeting in Helsinki."""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped


class WallsEmbed(eqx.Module):
    """A DeepSets model that extracts information about walls."""

    phi: eqx.nn.Sequential
    """The layer(s) applied to each wall in parallel."""
    rho: eqx.nn.MLP
    """The layer(s) applied to an permutation-invariant representation of all walls."""

    def __init__(
        self,
        intermediate_size: int = 500,
        out_size: int = 100,
        width_size: int = 500,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jax.random.split(key, 2)
        self.phi = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(jnp.ravel),
                eqx.nn.MLP(
                    in_size=4,
                    out_size=intermediate_size,
                    width_size=width_size,
                    depth=depth,
                    key=key1,
                ),
            ]
        )
        self.rho = eqx.nn.MLP(
            in_size=intermediate_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key2,
        )

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, walls: Float[Array, "num_walls 2 2"]
    ) -> Float[Array, "{self.mpl_rho.out_size}"]:
        x = jax.vmap(self.phi)(walls)
        x = jnp.sum(x, axis=0)
        x = self.rho(x)

        return x


class PathGenerator(eqx.Module):
    """A recurrent model that returns a path of a given order."""

    order: int
    cell: eqx.nn.GRUCell
    state2xy: eqx.nn.MLP

    def __init__(
        self,
        order: int,
        num_embeddings: int,
        hidden_size: int = 10,
        width_size: int = 100,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ):
        key1, key2 = jax.random.split(key, 2)

        self.order = order
        self.cell = eqx.nn.GRUCell(
            input_size=num_embeddings, hidden_size=hidden_size, key=key1
        )
        self.state2xy = eqx.nn.MLP(
            in_size=hidden_size,
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
        def scan_fn(state: tuple[Float[Array, ""], ...], _: None):
            state = self.cell(wall_embeddings)
            xy = self.state2xy(state)
            return self.cell(input_, state), xy

        init_state = x

        def scan_fn(state: tuple[Float[Array, ""], ...], _: None = None):
            state = self.cell(walls_embeddings)
            xy = ...
            self.cell(input_, state), state

        _, path = jax.lax.scan(scan_fn, init_state, length=self.order)

        return jnp.vstack((start, path, end))


class EvaluatePath(eqx.Module):
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
        scene_embeddings: Float[Array, " num_embeddings"],
    ) -> Float[Array, " "]:
        x = jnp.concatenate((jnp.ravel(path), scene_embeddings))
        x = self.mlp(x)

        return x


class Model(eqx.Module):
    """Global Deep-Learning model."""

    # hyperparameters
    order: int
    num_paths: int
    threshold: float
    inference: bool

    # trainable
    walls_embed: WallsEmbed
    path_generator: PathGenerator
    path_evaluator: PathEvaluator
    path_cell: eqx.nn.GRUCell

    @jax.jit
    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        order: int,
        num_paths: int,
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
        self.walls_embed = WallsEmbed(key=key1)
        self.path_generator = PathGenerator(order=order, key=key2)
        self.path_evaluator = PathEvaluator(order=order, key=key3)
        self.path_cell = eqx.nn.GRUCell

    def __call__(
        self, x: Float[Array, "2+num_walls*2 2"]
    ) -> Float[Array, "num_paths {self.order}+2 2"]:
        # Processing input
        tx = x[0, :]
        rx = x[1, :]
        walls = x[2:, :].reshape(-1, 2, 2)  # [num_walls, 2, 2]
        starts = walls[:, 0, :]
        ends = walls[:, 1, :]

        # todo: pass those as parameters to force using specular reflection
        directions, _ = jax.vmap(normalize)(ends - starts)
        normals = directions.at[:, 0].set(directions[:, 1])
        normals = normals.at[:, 1].set(-directions[:, 0])

        walls_embeddings = self.walls_emded(walls)

        # Generate paths

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

            while True:
                pass
        else:
            final_state, carries = jax.lax.scan(
                scan_fn, init_state, length=self.num_paths
            )

            paths, probabilities = carries

            return paths, probabilities

        # question: how to properly handle a variable-sized output?
        while True:
            path = self.gen_path(state, tx, rx)

            # question: how to combine both the path and the walls_embed
            # i.e., the knowledge we have about the geometry
            # do we just 'concat' the inputs altogether?
            is_valid = self.val_path(path)

            if is_valid < threshold:
                break

            paths.append(path)

            # state = update_state(self)
