
from typing import (
    List,
)
from dataclasses import (
    dataclass,
    field,
)


@dataclass(frozen=True)
class FPLCredentials:
    login: str
    password: str
    team_id: int


@dataclass(frozen=True, eq=True)
class Player:
    name: str = field(compare=True, hash=True)
    team: str = field(compare=True, hash=True)
    position: str = field(compare=True, hash=True)
    cost: int = field(compare=False, hash=False)
    score: float = field(compare=False, hash=False)
    points: int = field(compare=False, hash=False)


@dataclass(frozen=True, eq=True)
class Squad:
    players: List[Player]

    def score(self, acc=sum) -> float:
        return acc(p.score for p in self.players)

    def points(self, acc=sum) -> float:
        return acc(p.points for p in self.players)

    def cost(self, acc=sum) -> float:
        return acc(p.cost for p in self.players)
