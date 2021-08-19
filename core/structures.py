
from typing import (
    List,
)
from dataclasses import (
    dataclass,
)


@dataclass(frozen=True, eq=True)
class Player:
    name: str
    team: str
    position: str
    cost: int
    score: float
    points: int


@dataclass(frozen=True, eq=True)
class Squad:
    players: List[Player]

    def score(self, acc=sum) -> float:
        return acc(p.score for p in self.players)

    def points(self, acc=sum) -> float:
        return acc(p.points for p in self.players)

    def cost(self, acc=sum) -> float:
        return acc(p.cost for p in self.players)
