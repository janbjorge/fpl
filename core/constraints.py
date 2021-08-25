from typing import (
    List,
    Literal,
    Set,
)
from collections import (
    Counter,
)

from core.structures import Player


def team_constraint(lineup: List[Player], n=3):
    count = Counter(p.team for p in lineup)
    return max(count.values()) <= n


def position_constraint(
    lineup: List[Player],
    n: int,
    position: Literal["GKP", "DEF", "MID", "FWD"],
) -> bool:
    count = Counter(p.team for p in lineup if p.position == position)
    return max(count.values()) <= n


def gkp_def_not_same_team(
    lineup: List[Player],
) -> bool:
    _gkps = set(p.team for p in lineup if p.position == "GKP")
    _defs = set(p.team for p in lineup if p.position == "DEF")
    return not bool(_gkps.intersection(_defs))


def must_contain(
    lineup: List[Player],
    must: Set[str],
) -> bool:
    if must and lineup:
        return must.issubset(set(p.name for p in lineup))
    return True
