from typing import (
    Literal,
    Sequence,
    Set,
)
from collections import (
    Counter,
)

from core.structures import Player


def team_constraint(lineup: Sequence[Player], n=2):
    count = Counter(p.team for p in lineup)
    return max(count.values()) <= n


def position_constraint(
    lineup: Sequence[Player],
    n: int,
    position: Literal["GKP", "DEF", "MID", "FWD"],
) -> bool:
    count = Counter(p.team for p in lineup if p.position == position)
    return max(count.values()) <= n


def gkp_def_not_same_team(
    lineup: Sequence[Player],
) -> bool:
    _gkps = set(p.team for p in lineup if p.position == "GKP")
    _defs = set(p.team for p in lineup if p.position == "DEF")
    return not bool(_gkps.intersection(_defs))


def must_contain(
    lineup: Sequence[Player],
    must: Set[str],
) -> bool:
    if must and lineup:
        return must.issubset(set(p.name for p in lineup))
    return True


def valid_formation(lineup: Sequence[Player]) -> bool:

    if sum(1 for p in lineup if p.position == "GKP") != 1:
        return False

    if sum(1 for p in lineup if p.position == "DEF") < 3:
        return False

    if sum(1 for p in lineup if p.position == "FWD") < 1:
        return False

    return True
