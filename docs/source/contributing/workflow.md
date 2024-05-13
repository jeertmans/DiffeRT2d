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

As for every Python project,
using virtual environment is recommended to avoid conflicts between modules.
For DiffeRT2d,
we use
[Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).
If not already, please install it.

## Installing Python modules

With Poetry, installation becomes straightforward:

```bash
poetry install
```

Additionally, DiffeRT2d comes with group dependencies for development purposes:

```bash
poetry install --with dev  # For linters and formatters
# or
poetry install --with docs # To build the documentation locally
# or
poetry install --with test # To run tests
```

:::{note}
You can combine any number of groups when installing the package locally.
:::

## Running commands

As modules were installed in a new Python environment,
you cannot use them directly in the shell.
Instead, you either need to prepend `poetry run` to any command, e.g.:

```bash
poetry run python examples/plot_power_map.py
```

or enter a new shell that uses this new Python environment:

```bash
poetry shell
python examples/plot_power_map.py
```

## Testing your code

Most of the tests are done with GitHub actions, thus not on your computer.
The only command you should run locally is `pre-commit run --all-files`:
this runs a few linter and formatter to make sure the code quality and
style stay constant across time.
If a warning or an error is displayed, please fix it before going to next step.

However, testing your code locally can still be a good idea if you do not plan
to make a pull request soon. There are two test sets: those in the
`tests` folder, and the code in the docstring.

Pytest is used to run the former tests:

```bash
poetry run pytest
```

and `byexample` is used to run docstring code and check its output:

```bash
poetry run byexample -l python differt2d/scene.py
```

## Proposing changes

Once you feel ready and think your contribution is ready to be reviewed,
create a [pull request](https://github.com/jeertmans/DiffeRT2d/pulls)
and wait for a reviewer to check your work!

Many thanks to you!
