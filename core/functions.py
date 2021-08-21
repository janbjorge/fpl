from collections import (
    defaultdict,
)
from typing import (
    List,
)

import numpy as np

from core import (
    gather,
    helpers,
    structures,
)


def norm(values):
    _min = values.min()
    _max = values.max()
    return (values - _min) / (_max - _min)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lineup_cost(lineup, acc=sum):
    return acc(player.cost for player in lineup)


def lineup_score(lineup, acc=sum):
    return acc(player.score for player in lineup)


def xPtt(lineup: List[structures.Player], acc=sum) -> float:
    return acc(p.points for p in lineup)


def grp_by_cost(lineup):
    grp = defaultdict(list)
    for p in lineup:
        grp[p.cost].append(p)
    return grp


def top_n_score_by_cost(lineup, n=3):
    return [
        sorted(v, key=lambda v: v.score, reverse=True)[:n]
        for v in grp_by_cost(lineup).values()
    ]


def top_n_score_by_cost_by_positions(
    pool: List[structures.Player], cutoff=(2, 3, 3, 2)
):
    new = []
    for n, pos in zip(cutoff, gather.positions()):
        d = top_n_score_by_cost((p for p in pool if p.position == pos), n=n)
        new.extend(helpers.flatten(d))
    return new


def lprint(lineup: List[structures.Player]):
    for pos in gather.positions():
        pos_players = sorted(
            (p for p in lineup if p.position == pos),
            key=lambda p: p.score,
            reverse=True,
        )
        print(
            f"{pos}(n={len(pos_players)}, score={lineup_score(pos_players)}, cost={lineup_cost(pos_players)})"
        )
        for player in pos_players:
            print(f" {player}")


def sprint(pool):
    print(
        f"score: {lineup_score(pool)}, cost: {lineup_cost(pool)}, xP(TP): {xPtt(pool)}"
    )
    lprint(pool)


def tprint(old, new):

    print(
        f"old: score: {lineup_score(old)}, cost: {lineup_cost(old)}, xP(TP): {xPtt(old)}"
    )
    print(
        f"new: score: {lineup_score(new)}, cost: {lineup_cost(new)}, xP(TP): {xPtt(new)}"
    )

    change_old_new = sorted(
        set(old).difference(new), key=lambda n: (n.position, n.name)
    )
    change_new_old = sorted(
        set(new).difference(old), key=lambda n: (n.position, n.name)
    )

    for o, n in zip(change_old_new, change_new_old):
        print(f"{o} -> {n}")
