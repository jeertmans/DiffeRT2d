"""DiffeRT2d config tools."""
from contextlib import contextmanager

import jax


def varname(name: str) -> str:
    """
    Return the environ variable name for a give config variable name.

    :param name: The variable name.
    :return: The corresponding environ variable name.
    """
    return "DIFFERT2D_" + name.upper().replace(" ", "_").replace("-", "_")

class Config:
    def __init__(self) -> None:
        self.values = {}
        self.defaults = {}

    def get_env_or_default(self, name: str) -> bool:
        return jax.config.bool_env(varname(name), self.defaults[name])

    def __getattr__(self, name: str) -> bool:
        return self.values.setdefault(name, self.get_env_or_default(name))

    def __setattr__(self, name: str, value: bool) -> None:
        return self.values[name] = value

    def add_boolean_state(self, name: str, default: bool = False) -> ContextManager:
        self.defaults[name] = default

        @contextmanager
        def update(state: bool = default) -> Iterator[bool]:
            old_state = self.name
            try:
                self.name = state
                yield state
            finally:
                self.name = old_state

        return update


    name="jax_enable_approx",
    default=True,
    help=("Enable approximation using some activation function."),

enable_approx = Config
