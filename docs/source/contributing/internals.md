# Internals

DiffeRT2d's code is fully contained in the `differt2d` folder.

Each file's name tries to closely match its content, and nothing else.

For performance purposes, DiffeRT2d highly relies on JAX's just-in-time
compulition and array capabilities. As such, we use JAX arrays for most
of our primitive structures, and the `float32` type for computations that
are on real numbers.

## Docstring tests

When possible, a function should have some code example in its docstring, such
that it can also add as a test that everything works as expected by the user.

## More general tests

Less specific tests, or tests that require more configuration, are placed in
the `tests` folder. Those tests are also what is computed for the coverage of
this project.

You can read coverage details [here](https://app.codecov.io/gh/jeertmans/DiffeRT2d).
