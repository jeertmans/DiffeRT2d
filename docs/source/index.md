---
hide-toc: true
og:description: 2D Toolbox for Differentiable Ray Tracing
---

# Welcome to DiffeRT2d's documentation

DiffeRT2d is a Python toolbox for 2D differentiable Ray Tracing,
with a focus on Radio Propagation applications, where we are mostly
interested in simulating paths from one node (transmitter)
to another (receiver), i.e., Point-to-Point Ray Tracing.

```{eval-rst}
.. plot::
    :caption: Power map computed on a basic scene,
        with shadowing caused by walls.

    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from differt2d.scene import Scene
    from differt2d.utils import received_power

    ax = plt.gca()
    scene = Scene.basic_scene()
    X, Y = scene.grid(n=300)

    Z = scene.accumulate_on_receivers_grid_over_paths(
        X,
        Y,
        fun=received_power,
        reduce_all=True,
        approx=False,
    )
    ax.pcolormesh(X, Y, 10.0 * jnp.log10(Z))
    scene.plot(ax)
    plt.show()  # doctest: +SKIP
```


```{include} ../../README.md
:start-after: <!-- start description -->
:end-before: <!-- end description -->
```

The present tool is thoroughly documented, so please have a look at the
following sections:

```{toctree}
:titlesonly:
:maxdepth: 1

quickstart
examples_gallery/index
reference/index
research/index
jax_and_jaxtyping
references
```

If you are interested in contributing to this tool, please checkout the
[Contributing](contributing/index) section!

```{toctree}
:caption: Development
:hidden:

changelog
contributing/index
license
```
