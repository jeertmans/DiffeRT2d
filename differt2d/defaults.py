"""Default values."""

from typing import Literal

DEFAULT_ALPHA: float = 100.0
"""Default value for ``alpha`` parameter in
:func:`activation<differt2d.logic.activation>`."""

DEFAULT_FUNCTION: Literal["sigmoid", "hard_sigmoid"] = "hard_sigmoid"
"""Default value for ``function`` parameter in
:func:`activation<differt2d.logic.activation>`."""

DEFAULT_PATCH: float = 0.0
"""Default patch value applied to :meth:`Interactable.intersects_cartesian<differt2d.abc
.Interactable.intersects_cartesian>`."""
