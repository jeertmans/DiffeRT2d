"""Some useful type aliases."""

from typing import Union

from jaxtyping import Array, Float, Int

ArrayLikeFloat = Union[Float[Array, " *batch"], float]
ScalarFloat = Union[Float[Array, " "], float]
ScalarInt = Union[Int[Array, " "], int]
