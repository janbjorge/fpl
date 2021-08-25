import collections
import typing as T

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


def hyper_sigmoid(x):
    return 1 / (1 + np.exp(-(x - 0.5) * 10))


def caverge(
    samples: T.Sequence[T.Union[float, int]],
) -> T.Union[float, int]:
    # By applying this averger we pay more attion
    # to newer values than older values.
    weights = np.cos(np.linspace(0, 1, len(samples)) * np.pi / 3)
    weights /= weights.sum()
    return np.average(samples, weights=weights)


def lineup_cost(
    lineup: T.List[structures.Player],
    acc=sum,
):
    return round(acc(player.cost for player in lineup), 5)


def lineup_score(lineup: T.List[structures.Player], acc=sum):
    return round(acc(player.score for player in lineup), 5)


def xPtt(lineup: T.List[structures.Player], acc=sum) -> float:
    return round(acc(p.points for p in lineup), 5)


def grp_by_cost(
    lineup: T.List[structures.Player],
) -> T.DefaultDict[int, T.List[structures.Player]]:
    grp = collections.defaultdict(list)
    for p in lineup:
        grp[p.cost].append(p)
    return grp


def top_n_score_by_cost(
    lineup: T.List[structures.Player],
    n: int = 3,
) -> T.List[T.List[structures.Player]]:
    return [
        sorted(v, key=lambda v: v.score, reverse=True)[:n]
        for v in grp_by_cost(lineup).values()
    ]


def top_n_score_by_cost_by_positions(
    pool: T.List[structures.Player],
    cutoff: T.Tuple[int, int, int, int] = (2, 4, 4, 2),
) -> T.List[structures.Player]:
    new = []
    for n, pos in zip(cutoff, gather.positions()):
        d = top_n_score_by_cost([p for p in pool if p.position == pos], n=n)
        new.extend(helpers.flatten(d))
    return new


def lprint(lineup: T.List[structures.Player]) -> None:

    if not lineup:
        return

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


def sprint(pool: T.List[structures.Player]) -> None:

    if not pool:
        return

    print(
        f"score: {lineup_score(pool)}, cost: {lineup_cost(pool)}, xP(TP): {xPtt(pool)}"
    )
    lprint(pool)


def tprint(
    old: T.List[structures.Player],
    new: T.List[structures.Player],
) -> None:

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

    if not change_old_new or not change_new_old:
        return

    rs = len(str(max(change_old_new, key=lambda s: len(str(s)))))

    for o, n in zip(change_old_new, change_new_old):
        print(f"{str(o):<{rs}} => {str(n):>2}")
