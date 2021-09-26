import collections as C
import typing as T

from core import (
    structures,
)


def team_constraint(lineup: T.Sequence[structures.Player], n=2):
    count = C.Counter(p.team for p in lineup)
    return max(count.values()) <= n


def position_constraint(
    lineup: T.Sequence[structures.Player],
    n: int,
    position: T.Literal["GKP", "DEF", "MID", "FWD"],
) -> bool:
    count = C.Counter(p.team for p in lineup if p.position == position)
    return max(count.values()) <= n


def gkp_def_not_same_team(
    lineup: T.Sequence[structures.Player],
) -> bool:
    _gkps = set(p.team for p in lineup if p.position == "GKP")
    _defs = set(p.team for p in lineup if p.position == "DEF")
    return not bool(_gkps.intersection(_defs))


def must_contain(
    lineup: T.Sequence[structures.Player],
    must: T.Set[str],
) -> bool:
    if must and lineup:
        return must.issubset(set(p.name for p in lineup))
    return True


def valid_formation(lineup: T.Sequence[structures.Player]) -> bool:

    if sum(1 for p in lineup if p.position == "GKP") != 1:
        return False

    if sum(1 for p in lineup if p.position == "DEF") < 3:
        return False

    if sum(1 for p in lineup if p.position == "FWD") < 1:
        return False

    return True


def unique(lineup: T.Sequence[structures.Player]) -> bool:
    return len(set(p.name for p in lineup)) == len(lineup)
