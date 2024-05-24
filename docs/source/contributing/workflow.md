# Workflow

This document is there to help you recreate a working environment for DiffeRT2d.

## Dependencies

```{include} ../../../README.md
:start-after: <!-- start deps -->
:end-before: <!-- end deps -->
```

## Forking the repository and cloning it locally

We use GitHub to host DiffeRT2d's repository, and we encourage contributors to
use git in general.

Useful links:

* [GitHub's Hello World](https://docs.github.com/en/get-started/quickstart/hello-world).
* [GitHub Pull Request in 100 Seconds](https://www.youtube.com/watch?v=8lGpZkjnkt4&ab_channel=Fireship).

Once you feel comfortable with git and GitHub,
[fork](https://github.com/jeertmans/DiffeRT2d/fork)
the repository, and clone it locally.

As for every Python project, using virtual environment is recommended to avoid
conflicts between modules.
For this project, we use [Rye](https://rye.astral.sh/) to easily manage project
and development dependencies. If not already, please install this tool.

## Installing Python modules

With Rye, installation becomes straightforward:

```bash
rye sync --all-features
```

## Running commands

The following commands assume that you installed
the project locally with:

```bash
rye sync
```

and that you activated the corresponding Python virtual environment:

```bash
. .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

Then, you can run any Python script or command from the terminal, e.g.:

```bash
python examples/plot_power_map.py
```

## Testing your code

Most of the tests are done with GitHub actions, thus not on your computer.
The only command you should run locally is:

```bash
pre-commit run --all-files
```

This runs a few linter and formatter to make sure the code quality and style stay
constant across time.
If a warning or an error is displayed, please fix it before going to next step.

For testing your code, simply run:

```bash
pytest
```

## Building the documentation

The documentation is generated using Sphinx, based on the content
in `docs/source` and in the `differt2d` Python package.

To generate the documentation, run the following:

```bash
cd docs
make html
```

Then, the output index file is located at `docs/build/html/index.html` and
can be opened with any modern browser.

## Proposing changes

Once you feel ready and think your contribution is ready to be reviewed,
create a [pull request](https://github.com/jeertmans/DiffeRT2d/pulls)
and wait for a reviewer to check your work!

Many thanks to you!
