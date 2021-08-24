import argparse
import itertools
import shutil
import functools
import typing as T

from tqdm import (
    tqdm,
)

from core import (
    constraints,
    functions,
    gather,
    helpers,
    structures,
)


class FPLException(BaseException):
    pass


class InvalidLineup(FPLException):
    pass


def lineup(
    pool: T.List[structures.Player],
    verbose: bool = False,
    buget=1_000,
    base=(
        (),
        (),
        (),
        (),
    ),
) -> T.List[structures.Player]:

    _gkp = sorted(
        (p for p in pool if p.position == "GKP"), key=lambda p: p.score, reverse=True
    )
    _def = sorted(
        (p for p in pool if p.position == "DEF"), key=lambda p: p.score, reverse=True
    )
    _mid = sorted(
        (p for p in pool if p.position == "MID"), key=lambda p: p.score, reverse=True
    )
    _fwd = sorted(
        (p for p in pool if p.position == "FWD"), key=lambda p: p.score, reverse=True
    )

    m_gkps, m_defs, m_mids, m_fwds = [set(m) for m in base]

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
        sorted(_gkp_combinations(), key=functions.lineup_score, reverse=False)
    )
    def_combinations = tuple(
        sorted(_def_combinations(), key=functions.lineup_score, reverse=False)
    )
    mid_combinations = tuple(
        sorted(_mid_combinations(), key=functions.lineup_score, reverse=False)
    )
    fwd_combinations = tuple(
        sorted(_fwd_combinations(), key=functions.lineup_score, reverse=False)
    )

    min_cost_def = functions.lineup_cost(
        min(gkp_combinations, key=functions.lineup_cost)
    )
    min_cost_mid = functions.lineup_cost(
        min(mid_combinations, key=functions.lineup_cost)
    )
    min_cost_fwd = functions.lineup_cost(
        min(fwd_combinations, key=functions.lineup_cost)
    )

    max_cost_def = functions.lineup_cost(
        max(gkp_combinations, key=functions.lineup_cost)
    )
    max_cost_mid = functions.lineup_cost(
        max(mid_combinations, key=functions.lineup_cost)
    )
    max_cost_fwd = functions.lineup_cost(
        max(fwd_combinations, key=functions.lineup_cost)
    )

    max_score_gkp = functions.lineup_score(
        max(gkp_combinations, key=functions.lineup_score)
    )
    max_score_def = functions.lineup_score(
        max(def_combinations, key=functions.lineup_score)
    )
    max_score_mid = functions.lineup_score(
        max(mid_combinations, key=functions.lineup_score)
    )
    max_score_fwd = functions.lineup_score(
        max(fwd_combinations, key=functions.lineup_score)
    )

    min_cost_mid_fwd = min_cost_mid + min_cost_fwd
    min_cost_def_mid_fwd = min_cost_def + min_cost_mid + min_cost_fwd

    max_cost_mid_fwd = max_cost_mid + max_cost_fwd
    max_cost_def_mid_fwd = max_cost_def + max_cost_mid + max_cost_fwd

    max_score_mid_fwd = max_score_mid + max_score_fwd
    max_score_def_mid_fwd = max_score_def + max_score_mid + max_score_fwd

    best_lineup = list()
    buget_lower = buget * 0.9
    best_score = sum((max_score_gkp, max_score_def, max_score_mid, max_score_fwd)) * 0.9

    print(f"{min_cost_mid=}, {min_cost_fwd=}, {min_cost_mid_fwd=}")
    print(f"{max_score_mid=}, {max_score_fwd=}, {max_score_mid_fwd=}")
    print(f"{m_gkps=}, {m_defs=}, {m_mids=}, {m_fwds=}")

    def lvl0(c):
        return (
            buget_lower - max_cost_def_mid_fwd
            <= functions.lineup_cost(c)
            <= buget - min_cost_def_mid_fwd
            and functions.lineup_score(c) + max_score_def_mid_fwd > best_score
            and constraints.team_constraint(c)
        )

    def lvl1(c):
        return (
            buget_lower - max_cost_mid_fwd
            <= functions.lineup_cost(c)
            <= buget - min_cost_mid_fwd
            and functions.lineup_score(c) + max_score_mid_fwd > best_score
            and constraints.team_constraint(c)
            and constraints.gkp_def_not_same_team(c)
        )

    def lvl2(c):
        return (
            buget_lower - max_cost_fwd
            <= functions.lineup_cost(c)
            <= buget - min_cost_fwd
            and functions.lineup_score(c) + max_score_fwd > best_score
            and constraints.team_constraint(c)
        )

    def lvl3(c):
        return (
            functions.lineup_cost(c) <= buget
            and functions.lineup_score(c) > best_score
            and constraints.team_constraint(c)
        )

    total = (
        len(gkp_combinations)
        * len(def_combinations)
        * len(mid_combinations)
        * len(fwd_combinations)
    )
    step = len(mid_combinations) * len(fwd_combinations)

    with tqdm(
        total=total,
        bar_format="{percentage:3.0f}%|{bar:20}{r_bar}",
        unit_scale=True,
        unit_divisor=2 ** 10,
    ) as bar:
        for g in gkp_combinations:
            if lvl0(g):
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
                                        best_score = functions.lineup_score(g3)
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
) -> T.List[structures.Player]:

    old_lineup_cost = functions.lineup_cost(old)
    pool = list(set(pool) - set(old))

    pool = sorted(pool, key=lambda p: p.score, reverse=True)
    old = sorted(old, key=lambda p: p.score, reverse=True)

    @functools.lru_cache(maxsize=len(pool) * len(old))
    def tp(old, new):
        functions.tprint(old, new)
        print("-" * 100)

    def _transfers(
        current: T.List[structures.Player],
        best: T.List[structures.Player],
        n_transfers: int,
    ) -> T.List[structures.Player]:

        if n_transfers > max_transfers:
            raise InvalidLineup

        if n_transfers == max_transfers:
            if (
                functions.lineup_cost(current) <= old_lineup_cost
                and constraints.team_constraint(current)
                and constraints.gkp_def_not_same_team(current)
                and functions.lineup_score(current) > functions.lineup_score(best)
            ):
                return current
            raise InvalidLineup

        for transfer_in in pool:
            for idx, transfer_out in enumerate(best):

                if transfer_in.position != transfer_out.position:
                    continue

                # Transfer inn new player.
                tmp = current.copy()
                tmp[idx] = transfer_in

                try:
                    new = _transfers(tmp, best, n_transfers=n_transfers + 1)
                except InvalidLineup:
                    continue

                best = new.copy()
                if verbose:
                    if set(old) == set(best):
                        continue
                    tp(tuple(old), tuple(best))

        return best

    return _transfers(old, old, 0)


def argument_parser():
    parser = argparse.ArgumentParser(prog="Lazy FPL")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enables verbose mode.",
    )
    parser.add_argument(
        "-p",
        "--prune",
        action="store_true",
        help="Clear local cache",
    )

    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    transfer_parser = sub_parsers.add_parser(
        "transfer",
    )
    transfer_parser.add_argument(
        "max",
        type=int,
        nargs="?",
        default=2,
        help="Number of allowed transfers.",
    )

    lineup_parser = sub_parsers.add_parser(
        "lineup",
    )
    lineup_parser.add_argument(
        "-gkp",
        "--goalkeepers",
        nargs="+",
        help="Goalkeepers (FPL web-name) tha must be in the lineup.",
        default=[],
    )
    lineup_parser.add_argument(
        "-def",
        "--defenders",
        nargs="+",
        help="Defenders (FPL web-name) tha must be in the lineup.",
        default=[],
    )
    lineup_parser.add_argument(
        "-mid",
        "--midfielders",
        nargs="+",
        help="Midfielders (FPL web-name) tha must be in the lineup.",
        default=[],
    )
    lineup_parser.add_argument(
        "-fwd",
        "--forwards",
        nargs="+",
        help="Forwards (FPL web-name) tha must be in the lineup.",
        default=[],
    )

    print_parser = sub_parsers.add_parser(
        "print",
    )
    print_parser.add_argument(
        "show",
        nargs="?",
        choices=("team", "pool"),
        help="Print current FPL-team or player pool.",
    )

    return parser.parse_args()


def main():
    parsed = argument_parser()

    if parsed.prune and helpers.CACHE_FOLDER.exists():
        print(f"Clearing cache: {helpers.CACHE_FOLDER}")
        shutil.rmtree(helpers.CACHE_FOLDER)
        print(f"Clearing cache: {helpers.CACHE_FOLDER} - done")

    if parsed.mode == "transfer":
        old = gather.team()
        new = transfers(
            pool=functions.top_n_score_by_cost_by_positions(gather.player_pool()),
            old=old,
            max_transfers=parsed.max,
            verbose=parsed.verbose,
        )
        functions.tprint(old, new)

    elif parsed.mode == "lineup":
        functions.sprint(
            lineup(
                pool=functions.top_n_score_by_cost_by_positions(gather.player_pool()),
                verbose=parsed.verbose,
                base=(
                    tuple(parsed.goalkeepers),
                    tuple(parsed.defenders),
                    tuple(parsed.midfielders),
                    tuple(parsed.forwards),
                ),
            )
        )

    elif parsed.mode == "print":
        if parsed.show == "team":
            functions.sprint(gather.team())
        elif parsed.show == "pool":
            functions.lprint(
                functions.top_n_score_by_cost_by_positions(gather.player_pool()),
            )


if __name__ == "__main__":
    main()
