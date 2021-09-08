import dataclasses
import typing as T


@dataclasses.dataclass(frozen=True, eq=True)
class Strength:
    attack: int
    defence: int
    overall: int
    home: bool

    def mean(self) -> float:
        return (self.attack + self.defence + self.overall) / 3.0

    @staticmethod
    def from_dict(d: T.Dict) -> "Strength":
        return Strength(
            attack=d["attack"],
            defence=d["defence"],
            home=d["home"],
            overall=d["overall"],
        )

    def ratio(self, other: "Strength") -> float:
        if self.home:
            return self.mean() / other.mean()
        return other.mean() / self.mean()


@dataclasses.dataclass(frozen=True, eq=True)
class Player:
    name: str = dataclasses.field(compare=True, hash=True)
    team: str = dataclasses.field(compare=True, hash=True)
    position: str = dataclasses.field(compare=True, hash=True)
    cost: int
    points: int
    xP: float
