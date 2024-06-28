# DiffeRT2d

<img alt="DiffeRT2d Logo" align="right" width="512px" src="https://raw.githubusercontent.com/jeertmans/DiffeRT2d/main/static/logo_light_transparent.png">

[![Latest Release][pypi-version-badge]][pypi-version-url]
[![Python version][pypi-python-version-badge]][pypi-version-url]
[![Documentation][documentation-badge]][documentation-url]
[![DOI][doi-badge]][doi-url]
[![Codecov][codecov-badge]][codecov-url]

Differentiable Ray Tracing Python Framework for Radio Propagation.

<!-- start description -->

DiffeRT2d is built on top of the
[JAX](https://github.com/google/jax)
library to provide a program that is *differentiable everywhere*.
With that, performing gradient-based optimization, or training
Machine Learning models with Ray Tracing (RT) becomes straightforward!
Moreover, the extensive use of the object-oriented paradigm
facilitates the simulation of complex objects, such as metasurfaces,
and the use of more advanced path tracing methods.

The objective of this tool is to provide a **simple-to-use** and
**highly interpretable** RT framework **for researchers** engaged
in fundamental studies of RT applied to radio propagation,
or any researcher interested in the various paths radio waves
can take in a given environment.

<!-- end description -->

> [!IMPORTANT]
> For 3D scenarios at city-scales,
> checkout [DiffeRT](https://github.com/jeertmans/DiffeRT).

- [Installation](#installation)
  * [Dependencies](#dependencies)
  * [Pip install](#pip-install)
  * [Install From Repository](#install-from-repository)
- [Usage](#usage)
- [Contributing](#contributing)
  * [Reporting an Issue](#reporting-an-issue)
  * [Seeking for Help](#seeking-for-help)
  * [Contact](#contact)

## Installation

<!-- start install -->

While installing DiffeRT2d and its dependencies on your global Python is fine,
we recommend using a virtual environment
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

The recommended way to install the latest release is to use pip:

```bash
pip install differt2d
```

### Install From Repository

An alternative way to install DiffeRT2d is to clone the git repository,
and install from there:
read the
[contributing guide](https://eertmans.be/DiffeRT2d/latest/contributing/workflow.html)
to know how.

<!-- end install -->

## Usage

For a quick introduction to DiffeRT2d, check you our
[Quickstart](https://eertmans.be/DiffeRT2d/latest/quickstart.html) tutorial!

You may find a multitude of usage examples across the documentation
or the [examples](https://github.com/jeertmans/DiffeRT2d/tree/main/examples)
folder, or directly in the
[examples gallery](https://eertmans.be/DiffeRT2d/latest/examples_gallery/index.html).

## Contributing

Contributions are more than welcome!
Please read through
[our contributing section](https://eertmans.be/DiffeRT2d/latest/contributing/index.html).

### Reporting an Issue

<!-- start reporting-an-issue -->

If you think you found a bug,
an error in the documentation,
or wish there was some feature that is currently missing,
we would love to hear from you!

The best way to reach us is via the
[GitHub issues](https://github.com/jeertmans/DiffeRT2d/issues).
If your problem is not covered by an already existing (closed or open) issue,
then we suggest you create a
[new issue](https://github.com/jeertmans/DiffeRT2d/issues/new).

The more precise you are in the description of your problem, the faster we will
be able to help you!

<!-- end reporting-an-issue -->

### Seeking for help

<!-- start seeking-for-help -->

Sometimes, you may have a question about ,
not necessarily an issue.

There are two ways you can reach us for questions:

- via the
[GitHub issues](https://github.com/jeertmans/DiffeRT2d/issues);
- or via
[GitHub discussions](https://github.com/jeertmans/DiffeRT2d/discussions).

<!-- end seeking-for-help -->

### Contact

<!-- start contact -->

Finally, if you do not have any GitHub account,
or just wish to contact the author of DiffeRT2d,
you can do so at: [jeertmans@icloud.com](mailto:jeertmans@icloud.com).

<!-- end contact -->

[pypi-version-badge]: https://img.shields.io/pypi/v/DiffeRT2d?label=DiffeRT2d
[pypi-version-url]: https://pypi.org/project/DiffeRT2d/
[pypi-python-version-badge]: https://img.shields.io/pypi/pyversions/DiffeRT2d
[documentation-badge]: https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=documentation&up_color=green&up_message=online&url=https%3A%2F%2Feertmans.be%2FDiffeRT2d%2F
[documentation-url]: https://eertmans.be/DiffeRT2d/
[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.11386517.svg
[doi-url]: https://doi.org/10.5281/zenodo.11386517
[codecov-badge]: https://codecov.io/gh/jeertmans/DiffeRT2d/branch/main/graph/badge.svg?token=1dJ1AKWMR5
[codecov-url]: https://codecov.io/gh/jeertmans/DiffeRT2d
