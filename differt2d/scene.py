import chex

from .abc import Interaction
from .typing import List


@chex.dataclass
class Scene:
    objects: List[Interaction]
