from dataclasses import (
    dataclass,
    field,
)

from typing import (
    List,
)


@dataclass(frozen=True, eq=True)
class Player:
    name: str = field(compare=True, hash=True)
    team: str = field(compare=True, hash=True)
    position: str = field(compare=True, hash=True)
    cost: int = field(compare=False, hash=False)
    score: float = field(compare=False, hash=False)
    points: int = field(compare=False, hash=False)
