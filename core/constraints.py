
from typing import (
    List,
    Literal,
    Set,
)
from collections import (
    Counter,
)

from core.structures import Player


def team_constraint(lineup: List[Player], n=2):
    count = Counter(p.team for p in lineup)
    return max(count.values()) <= n


def position_constraint(
    lineup: List[Player],
    n: int,
    position: Literal['GKP', 'DEF', 'MID', 'FWD'],
) -> bool:
    count = Counter(p.team for p in lineup if p.position == position)
    return max(count.values()) <= n


def gkp_def_not_same_team(
    lineup: List[Player],
) -> bool:
    count = Counter(p.team for p in lineup if p.position == 'GKP' or p.position == 'DEF')
    return max(count.values()) <= 1


def must_contain(
    lineup: List[Player],
    must: Set[str],
) -> bool:
    if must and lineup:
        return must.issubset(set(p.name for p in lineup))
    return True


def team_is_2_5_5_3(lineup: List[Player]) -> bool:
    count = Counter(p.position for p in lineup)
    return (
        count['GKP'] == 2 and
        count['DEF'] == 5 and
        count['MID'] == 5 and
        count['FWD'] == 3
    )
