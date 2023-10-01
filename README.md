<img alt="DiffeRT2d Logo" align="right" width="512px" src="https://raw.githubusercontent.com/jeertmans/DiffeRT2d/main/static/logo_light_transparent.png">

# DiffeRT2d

[![Documentation][documentation-badge]][documentation-url]
[![codecov][codecov-badge]][codecov-url]

Differentiable Ray Tracing (RT) Python framework for Telecommunications-oriented applications.

> **NOTE**: the present work offers a simple Python module to create basic 2D scenarios,
> and should be used for experimental purposes only.

- [Installation](#installation)
  * [Dependencies](#dependencies)
  * [Pip install](#pip-install)
  * [Install From Repository](#install-from-repository)
- [Usage](#usage)

## Installation

<!-- start install -->

While installing DiffeRT2D and its dependencies on your global Python is fine,
I recommend using a virtual environment
(e.g., [venv](https://docs.python.org/3/tutorial/venv.html)) for a local installation.

### Dependencies

<!-- start deps -->

DiffeRT2d uses [JAX](https://github.com/google/jax)
for automatic differentation,
which in turn may use (or not) CUDA for GPU acceleration.

If needed, please refer to
[JAX's installation guidelines](https://github.com/google/jax#installation)
for more details.

<!-- end deps -->

### Pip Install

<!-- TODO -->

The recommended way to install the latest release is to use pip:

```bash
pip install differt2d
```

### Install From Repository

<!-- TODO -->

An alternative way to install DiffeRT2d is to clone the git repository,
and install from there:
read the
[contributing guide](https://eertmans.be/DiffeRT2d/contributing/workflow.html)
to know how.

<!-- end install -->

## Usage

<!-- start usage -->

You may find a multitude of usage examples across the documentation
or the [examples](https://github.com/jeertmans/DiffeRT2d/tree/main/examples)
folder, or directly in the
[examples gallery](https://eertmans.be/DiffeRT2d/examples_gallery/index.html).

<!-- end usage -->

[documentation-badge]: https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=documentation&up_color=green&up_message=online&url=https%3A%2F%2Feertmans.be%2FDiffeRT2d%2F
[documentation-url]: https://eertmans.be/DiffeRT2d/
[codecov-badge]: https://codecov.io/gh/jeertmans/DiffeRT2d/branch/main/graph/badge.svg?token=1dJ1AKWMR5
[codecov-url]: https://codecov.io/gh/jeertmans/DiffeRT2d
