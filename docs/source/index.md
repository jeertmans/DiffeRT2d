---
hide-toc: true
og:description: 2D Toolbox for Differentiable Ray Tracing
---

# Welcome to DiffeRT2d's documentation

DiffeRT2D is a Python toolbox for 2D differentiable Ray Tracing,
with a focus on telecommunications-oriented applications[^1].

[^1]: This means we are mostly interesting in simulating paths from
  one node (emitter) to another (receiver),
  and not a highly efficient image renderer, like in computer gaphics.

The present tool is thoroughly documented, so please have a look at the
following sections:

```{toctree}
:titlesonly:
:maxdepth: 1

quickstart
examples_gallery/index
reference/index
```

If you are intersted in contributing to this tool, please checkout the
[Contributing](contributing/index) section!

```{eval-rst}
.. plot::
    :caption: Power map computed on a basic scene,
        with shadowing caused by object.

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
        reduce=True,
        approx=False,
    )
    ax.pcolormesh(X, Y, 10.0 * jnp.log10(Z))
    scene.plot(ax)
    plt.show()
```

```{toctree}
:caption: Development
:hidden:

changelog
contributing/index
license
```
