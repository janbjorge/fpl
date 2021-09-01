import dataclasses
import typing as T

from core import (
    helpers,
)

_POSITIONS = T.Literal["GKP", "DEF", "MID", "FWD"]

NUMBER = T.Union[int, float]
POSITIONS = T.get_args(_POSITIONS)


@dataclasses.dataclass(frozen=False)
class Samples:

    historical: T.List[NUMBER] = dataclasses.field(default_factory=list)
    future: T.List[NUMBER] = dataclasses.field(default_factory=list)

    caverge_historical: float = dataclasses.field(init=False)
    caverge_future: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.caverge_historical = helpers.caverge(self.historical)
        self.caverge_future = helpers.caverge(self.future)


@dataclasses.dataclass(frozen=True, eq=True)
class Player:
    name: str = dataclasses.field(compare=True, hash=True)
    team: str = dataclasses.field(compare=True, hash=True)
    position: _POSITIONS = dataclasses.field(compare=True, hash=True)
    cost: int
    points: int
    xP: float


@dataclasses.dataclass(frozen=True)
class TeamStrength:
    name: str
    attack: int
    defence: int
    overall: int

@dataclasses.dataclass(frozen=True)
class RStrength:
    attack: float
    defence: float
    overall: float


@dataclasses.dataclass(frozen=True)
class TPlayer:
    name: str = dataclasses.field(compare=True, hash=True)
    position: _POSITIONS
    points_gained: int


@dataclasses.dataclass(frozen=True)
class TrainingObservation:
    gw: int
    player: TPlayer
    RStrength: RStrength
