import functools
import json
import pathlib
import statistics
import typing as T

import pandas as pd

from core import (
    gather,
    structures,
)


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


def remove_bad(
    pool: T.List[structures.Player],
    min_xp: T.Optional[float],
    must: T.Set[str],
) -> T.List[structures.Player]:

    cutoff = {}
    for pos in set(p.position for p in pool):
        if min_xp is None:
            cutoff[pos] = statistics.variance([p.xP for p in pool if p.position == pos])
        else:
            cutoff[pos] = min_xp

    return [p for p in pool if p.xP >= cutoff[p.position] or p.name in must]


def summary(
    samples: T.Sequence[float],
    functions=(
        statistics.fmean,
        statistics.mean,
        statistics.median_grouped,
        statistics.median_high,
        statistics.median_low,
        statistics.median,
        statistics.mode,
        statistics.pstdev,
        statistics.pvariance,
        statistics.stdev,
        statistics.variance,
    ),
) -> None:
    rs = max(len(f.__name__) for f in functions)
    for f in functions:
        print(f"{f.__name__:<{rs}} : {round(f(samples),1):>2}")


@functools.lru_cache(maxsize=None)
def cached_csv_read(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)


@functools.lru_cache(maxsize=None)
def teams_gw_merge(
    teams: pathlib.Path,
    gw: pathlib.Path,
) -> pd.DataFrame:

    _teams = cached_csv_read(teams)
    _gw = cached_csv_read(gw)

    merged = _gw.merge(_teams, left_on="opponent_team", right_on="id")
    merged.sort_values("GW", inplace=True, ascending=False)
    merged["web_name"] = merged.name_x.apply(gather.fullname_to_web_name)
    merged = merged[merged["web_name"].notna()]
    merged.reset_index(inplace=True)
    return merged
