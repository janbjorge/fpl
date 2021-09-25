import functools
import itertools
import typing as T

from tqdm import (
    tqdm,
)

from core import (
    constraints,
    exceptions,
    functions,
    gather,
    structures,
)


def lineup(
    pool: T.List[structures.Player],
    verbose: bool = False,
    buget=1_000,
    ignore: T.Tuple[str, ...] = tuple(),
    base=(
        (),
        (),
        (),
        (),
    ),
) -> T.List[structures.Player]:

    pool = [p for p in pool if p.name not in ignore]

    _gkp = sorted(
        (p for p in pool if p.position == "GKP"), key=lambda p: p.xP, reverse=True
    )
    _def = sorted(
        (p for p in pool if p.position == "DEF"), key=lambda p: p.xP, reverse=True
    )
    _mid = sorted(
        (p for p in pool if p.position == "MID"), key=lambda p: p.xP, reverse=True
    )
    _fwd = sorted(
        (p for p in pool if p.position == "FWD"), key=lambda p: p.xP, reverse=True
    )

    m_gkps, m_defs, m_mids, m_fwds = tuple(set(m) for m in base)

    for m_gkp in m_gkps:
        if m_gkp not in set(g.name for g in _gkp):
            print(f"Unkown goalkeeper: {m_gkp}")

    for m_def in m_defs:
        if m_def not in set(d.name for d in _def):
            print(f"Unkown defender: {m_def}")

    for m_mid in m_mids:
        if m_mid not in set(m.name for m in _mid):
            print(f"Unkown midfielder: {m_mid}")

    for m_fwd in m_fwds:
        if m_fwd not in set(f.name for f in _fwd):
            print(f"Unkown forwarder {m_fwd}")

    def _gkp_combinations():
        yield from filter(
            lambda g: constraints.position_constraint(g, 1, "GKP")
            and constraints.must_contain(g, m_gkps),
            itertools.combinations(_gkp, 2),
        )

    def _def_combinations():
        yield from filter(
            lambda d: constraints.position_constraint(d, 1, "DEF")
            and constraints.must_contain(d, m_defs),
            itertools.combinations(_def, 5),
        )

    def _mid_combinations():
        yield from filter(
            lambda m: constraints.position_constraint(m, 2, "MID")
            and constraints.must_contain(m, m_mids),
            itertools.combinations(_mid, 5),
        )

    def _fwd_combinations():
        yield from filter(
            lambda f: constraints.position_constraint(f, 1, "FWD")
            and constraints.must_contain(f, m_fwds),
            itertools.combinations(_fwd, 3),
        )

    gkp_combinations = tuple(
        sorted(_gkp_combinations(), key=functions.lineup_xp, reverse=True)
    )
    def_combinations = tuple(
        sorted(_def_combinations(), key=functions.lineup_xp, reverse=True)
    )
    mid_combinations = tuple(
        sorted(_mid_combinations(), key=functions.lineup_xp, reverse=True)
    )
    fwd_combinations = tuple(
        sorted(_fwd_combinations(), key=functions.lineup_xp, reverse=True)
    )

    total = (
        len(gkp_combinations)
        * len(def_combinations)
        * len(mid_combinations)
        * len(fwd_combinations)
    )

    print(f"Goalkeeper combinations: {len(gkp_combinations)}")
    print(f"Defender combinations:   {len(def_combinations)}")
    print(f"Midfielder combinations: {len(mid_combinations)}")
    print(f"Forwarder combinations:  {len(fwd_combinations)}")
    print(f"Total combinations:      {total:.1e}")

    if not total:
        return []

    min_cost_mid = functions.lineup_cost(
        min(mid_combinations, key=functions.lineup_cost)
    )
    min_cost_fwd = functions.lineup_cost(
        min(fwd_combinations, key=functions.lineup_cost)
    )

    max_cost_mid = functions.lineup_cost(
        max(mid_combinations, key=functions.lineup_cost)
    )
    max_cost_fwd = functions.lineup_cost(
        max(fwd_combinations, key=functions.lineup_cost)
    )

    max_xp_gkp = functions.lineup_xp(max(gkp_combinations, key=functions.lineup_xp))
    max_xp_def = functions.lineup_xp(max(def_combinations, key=functions.lineup_xp))
    max_xp_mid = functions.lineup_xp(max(mid_combinations, key=functions.lineup_xp))
    max_xp_fwd = functions.lineup_xp(max(fwd_combinations, key=functions.lineup_xp))

    min_cost_mid_fwd = min_cost_mid + min_cost_fwd
    max_cost_mid_fwd = max_cost_mid + max_cost_fwd
    max_xp_mid_fwd = max_xp_mid + max_xp_fwd

    best_lineup: T.List[structures.Player] = []
    buget_lower = buget * 0.95
    best_xp = sum((max_xp_gkp, max_xp_def, max_xp_mid, max_xp_fwd))
    step = len(mid_combinations) * len(fwd_combinations)

    if verbose:
        print(f"{min_cost_mid=}, {min_cost_fwd=}, {min_cost_mid_fwd=}")
        print(f"{max_xp_mid=}, {max_xp_fwd=}, {max_xp_mid_fwd=}")
        print(f"{m_gkps=}, {m_defs=}, {m_mids=}, {m_fwds=}")

    def lvl1(c):
        return (
            buget_lower - max_cost_mid_fwd
            <= functions.lineup_cost(c)
            <= buget - min_cost_mid_fwd
            and functions.lineup_xp(c) + max_xp_mid_fwd > best_xp
            and constraints.team_constraint(c)
            and constraints.gkp_def_not_same_team(c)
        )

    def lvl2(c):
        return (
            buget_lower - max_cost_fwd
            <= functions.lineup_cost(c)
            <= buget - min_cost_fwd
            and functions.lineup_xp(c) + max_xp_fwd > best_xp
            and constraints.team_constraint(c)
        )

    def lvl3(c):
        return (
            functions.lineup_cost(c) <= buget
            and functions.lineup_xp(c) > best_xp
            and constraints.team_constraint(c)
        )

    while not best_lineup:

        best_xp = best_xp * 0.95

        if verbose:
            print(f"best_xp={best_xp}")

        with tqdm(
            total=total,
            bar_format="{percentage:3.0f}%|{bar:20}{r_bar}",
            unit_scale=True,
            unit_divisor=2 ** 10,
        ) as bar:
            for g in gkp_combinations:
                for d in def_combinations:
                    bar.update(step)
                    g1 = g + d
                    if lvl1(g1):
                        for m in mid_combinations:
                            g2 = g1 + m
                            if lvl2(g2):
                                for f in fwd_combinations:
                                    g3 = g2 + f
                                    if lvl3(g3):
                                        best_xp = functions.lineup_xp(g3)
                                        best_lineup = g3
                                        if verbose:
                                            print("-" * 100)
                                            functions.sprint(g3)

    return best_lineup


def transfers(
    pool: T.List[structures.Player],
    old: T.List[structures.Player],
    max_transfers: int,
    verbose: bool = False,
    add: T.Tuple[str, ...] = tuple(),
    remove: T.Tuple[str, ...] = tuple(),
    ignore: T.Tuple[str, ...] = tuple(),
) -> T.List[structures.Player]:

    old_lineup_cost = functions.lineup_cost(old)
    pool = list(set(pool) - set(old))

    pool = sorted(pool, key=lambda p: p.xP, reverse=False)
    old = sorted(old, key=lambda p: p.xP, reverse=False)

    @functools.lru_cache(maxsize=len(pool) * len(old))
    def tp(old, new):
        functions.tprint(old, new)
        print("-" * 100)

    def valid_add(lineup: T.List[structures.Player]) -> bool:
        if not add:
            return True
        names = set(p.name for p in lineup)
        return all(a in names for a in add)

    def valid_remove(lineup: T.List[structures.Player]) -> bool:
        if not remove:
            return True
        names = set(p.name for p in lineup)
        return all(r not in names for r in remove)

    pool = [p for p in pool if p.name not in ignore]

    for p in add:
        if not any(l.name == p for l in pool):
            print(f"Player '{p}' not in seclection pool.")

    for p in remove:
        if not any(l.name == p for l in old):
            print(f"Player '{p}' not in current lineup.")

    def _transfers(
        current: T.List[structures.Player],
        best: T.List[structures.Player],
        n_transfers: int,
    ) -> T.List[structures.Player]:

        if n_transfers > max_transfers:
            raise exceptions.InvalidLineup

        if n_transfers == max_transfers:
            if (
                functions.lineup_cost(current) <= old_lineup_cost
                and constraints.team_constraint(current)
                and constraints.gkp_def_not_same_team(current)
                and functions.lineup_xp(current) > functions.lineup_xp(best)
                and valid_add(current)
                and valid_remove(current)
            ):
                return current
            raise exceptions.InvalidLineup

        if n_transfers == 0:
            _pool = tqdm(pool)
        else:
            _pool = pool

        for transfer_in in _pool:
            for idx, transfer_out in enumerate(best):

                if transfer_in.position != transfer_out.position:
                    continue

                # Transfer inn new player.
                tmp = current.copy()
                tmp[idx] = transfer_in

                try:
                    best = _transfers(tmp, best, n_transfers=n_transfers + 1)
                except exceptions.InvalidLineup:
                    continue

                if verbose:
                    if set(old) == set(best):
                        continue
                    tp(tuple(old), tuple(best))
        return best

    return _transfers(old, old, 0)


def gmw_lineup(
    current: T.List[structures.Player],
    gmw: T.Optional[int] = None,
    size: int = 11,
) -> T.List[structures.Player]:

    if gmw is None:
        gmw = gather.current_gameweek()

    lineups = itertools.combinations(
        current,
        size,
    )

    best_xp = 0.0
    best_lineup = []

    for lineup in lineups:
        xp = functions.lineup_xp(lineup)
        if constraints.valid_formation(lineup) and xp > best_xp:
            best_xp = xp
            best_lineup = list(lineup)

    return best_lineup
