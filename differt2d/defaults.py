"""Default values."""

DEFAULT_ALPHA: float = 100.0
"""Default value for ``alpha`` parameter in
:func:`activation<differt2d.logic.activation>`."""

DEFAULT_PATCH: float = 0.0
"""Default patch value applied to
:meth:`Interactable.intersects_cartesian<differt2d.abc
.Interactable.intersects_cartesian>`."""

DEFAULT_R_COEF: float = 0.5
"""Default value for real reflection coefficient."""

DEFAULT_HEIGHT: float = 0.1
"""Default TX antenna height, used to avoid division by zero when computing
:func:`received_power<differt2d.utils.received_power>`."""
