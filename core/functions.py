import collections
import typing as T

import numpy as np

from core import (
    gather,
    helpers,
    structures,
)


def caverge(
    samples: T.Sequence[T.Union[float, int]],
) -> T.Union[float, int]:
    # By applying this averger we pay more attion
    # to newer values than older values.
    if not samples:
        return 0
    weights = np.cos(np.linspace(0, 1, len(samples)) * np.pi / 3)
    weights /= weights.sum()
    return np.average(samples, weights=weights)


def lineup_cost(
    lineup: T.List[structures.Player],
    acc=sum,
):
    return round(acc(player.cost for player in lineup), 5)


def lineup_xp(
    lineup: T.Sequence[structures.Player],
    acc=sum,
) -> float:
    return round(acc(p.xP for p in lineup), 1)


def lineup_tp(lineup: T.Sequence[structures.Player], acc=sum) -> float:
    return round(acc(p.points for p in lineup), 1)


def grp_by_cost(
    pool: T.List[structures.Player],
) -> T.DefaultDict[int, T.List[structures.Player]]:
    grp = collections.defaultdict(list)
    for p in pool:
        grp[p.cost].append(p)
    return grp


def top_n_xp_by_cost(
    pool: T.List[structures.Player],
    n,
) -> T.List[T.List[structures.Player]]:
    return [
        sorted(v, key=lambda v: v.xP, reverse=True)[:n]
        for v in grp_by_cost(pool).values()
    ]


def top_n_xp_by_cost_by_positions(
    pool: T.List[structures.Player],
    cutoff: T.Tuple[int, int, int, int] = (2, 3, 3, 2),
) -> T.List[structures.Player]:
    new = []
    for n, pos in zip(cutoff, gather.positions()):
        d = top_n_xp_by_cost([p for p in pool if p.position == pos], n=n)
        new.extend(helpers.flatten(d))
    return new


def lprint(lineup: T.List[structures.Player]) -> None:

    if not lineup:
        return

    for pos in gather.positions():
        pos_players = sorted(
            (p for p in lineup if p.position == pos),
            key=lambda p: p.xP,
            reverse=True,
        )
        header(pos_players, prefix=f"{pos}(n={len(pos_players)}, ", postfix=")")
        for player in pos_players:
            print(f" {player}")


def header(pool: T.List[structures.Player], prefix="", postfix="") -> None:
    print(
        f"{prefix}cost: {lineup_cost(pool)}, TP: {lineup_tp(pool)}, xP: {lineup_xp(pool)}{postfix}"
    )


def sprint(pool: T.List[structures.Player]) -> None:
    if not pool:
        return
    header(pool)
    lprint(pool)


def tprint(
    old: T.List[structures.Player],
    new: T.List[structures.Player],
) -> None:

    change_old_new = sorted(
        set(old).difference(new), key=lambda n: (n.position, n.name)
    )
    change_new_old = sorted(
        set(new).difference(old), key=lambda n: (n.position, n.name)
    )

    if not change_old_new or not change_new_old:
        return

    header(old, prefix="old: ")
    header(new, prefix="new: ")

    rs = len(str(max(change_old_new, key=lambda s: len(str(s)))))

    for o, n in zip(change_old_new, change_new_old):
        print(f"{str(o):<{rs}} => {str(n):>2}")


def xP(
    tp: int,
    opponents: structures.Samples,
    gmw: int,
) -> float:
    return round(
        (opponents.caverge_historical * tp) / (opponents.caverge_future * gmw), 1
    )
