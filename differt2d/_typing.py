"""Some useful type aliases."""

from typing import Union

from jaxtyping import Array, Float, UInt

ArrayLikeFloat = Union[Float[Array, " *batch"], float]
ScalarFloat = Union[Float[Array, " "], float]
ScalarUInt = Union[UInt[Array, " "], int]  # TODO: annotate that int must be >= 0.
