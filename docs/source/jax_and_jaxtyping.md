# JAX and jaxtyping

This library would be nothing without the [JAX](https://github.com/google/jax)
library. JAX provides both automatic differentiation (*autodiff*)
and accelerated algebra (*XLA*) in the same Python package.

Because we use JAX arrays everywhere in the code,
you can evaluate the gradient of nearly any function
by wrapping it inside {func}`jax.grad`, e.g.:

```python
import jax
import jax.numpy as jnp
from differt2d.logic import activation

x = jnp.linspace(-5, +5)

dfdx = jax.grad(activation)(x)  # Voilà!
```

Moreover, JAX also provides just-in-time compilation
so your function runs even faster the next time you call it!

Most functions in DiffeRT2d are already wrapped with
{func}`jax.jit` so you do not have to care about
wrapping them.

## Installing JAX

Depending on whether you want to execute your code on CPUs
or GPUs, the installation of JAX is different.

If you just need the CPU version,
`pip install differt2d` should do the job.

Please look at their
[installation guide](https://github.com/google/jax#installation)
for more details.

## Understanding JAX Arrays

The main advantage of JAX over other *autodiff* libraries
(like PyTorch or TensorFlow) is that you can pretty much use
it as a drop-in replacement for [NumPy](https://numpy.org/),
because JAX contains a NumPy-compatible implementation of almost
all NumPy functions in {mod}`jax.numpy`.

JAX also comes with the concept of
[PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html),
and JAX arrays are PyTrees.

The only thing you **really need to know** is that PyTrees
are **immutable**. Hence, for compatibility with JAX's philosophy,
DiffeRT2d's objects are also **immutable PyTrees**.

As such, every method these objects expose is likely to return
a **new object instance**, e.g.:

```python
from differt.geometry import Point
from differt.scene import Scene

scene = Scene()

scene.with_transmitters(tx=Point())  # Don't

scene = scene.with_transmitters(tx=Point())  # Do
```

Finally, because our objects are PyTrees,
you can use {func}`equinox.tree_at` to *transform* a PyTree,
i.e., returning a new object after the transformation.

## Type-checking JAX Arrays

By design, Python does not care about type hints and this
is usually quite hard to enforce that a specific input array
must be, e.g., two-dimensional.

With [`jaxtyping`](https://docs.kidger.site/jaxtyping/),
we provide both meaningful type hints for every method in our API,
as well as enforce strong type and dimensionality checking at runtime.
Thanks to just-in-time compilation, the overhead of runtime type
checking is minimal and only performed once for each compiled version
of a given function.

For example, the following code illustrates that
function `my_function` takes a 2D array of floating-point values, `x`,
and returns a 1D array whose length matches the first axis of
the input array `x`.

```python
import jax
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


@jax.jit
@jaxtyped(typechecker=typechecker)
def my_function(x: Float[Array, "m n"]) -> Float[Array, " m"]:
    return jnp.mean(x, axis=0)
```
