
from typing import (
    List,
    Tuple,
)
from itertools import (
    combinations,
)

from tqdm import (
    tqdm,
)

from core import (
    constraints,
    functions,
    gather,
    structures,
)


class FPLException(BaseException):
    pass


class InvalidLineup(FPLException):
    pass


def lineup(
    pool: List[structures.Player],
    buget=1_000,
    base=((), (), (), (),),
) -> List[structures.Player]:

    _gkp = sorted((p for p in pool if p.position == 'GKP'), key=lambda p: p.score, reverse=True)
    _def = sorted((p for p in pool if p.position == 'DEF'), key=lambda p: p.score, reverse=True)
    _mid = sorted((p for p in pool if p.position == 'MID'), key=lambda p: p.score, reverse=True)
    _fwd = sorted((p for p in pool if p.position == 'FWD'), key=lambda p: p.score, reverse=True)

    m_gkps, m_defs, m_mids, m_fwds = [set(m) for m in base]

    def _gkp_combinations():
         yield from filter(
            lambda g: constraints.position_constraint(g, 1, 'GKP') and constraints.must_contain(g, m_gkps),
            combinations(_gkp, 2)
        )

    def _def_combinations():
         yield from filter(
            lambda d: constraints.position_constraint(d, 1, 'DEF') and constraints.must_contain(d, m_defs),
            combinations(_def, 5)
        )

    def _mid_combinations():
         yield from filter(
            lambda m: constraints.position_constraint(m, 2, 'MID') and constraints.must_contain(m, m_mids),
            combinations(_mid, 5)
        )

    def _fwd_combinations():
        yield from filter(
            lambda f: constraints.position_constraint(f, 1, 'FWD') and constraints.must_contain(f, m_fwds),
            combinations(_fwd, 3)
        )

    gkp_combinations = tuple(sorted(_gkp_combinations(), key=functions.lineup_score, reverse=False))
    def_combinations = tuple(sorted(_def_combinations(), key=functions.lineup_score, reverse=False))
    mid_combinations = tuple(sorted(_mid_combinations(), key=functions.lineup_score, reverse=False))
    fwd_combinations = tuple(sorted(_fwd_combinations(), key=functions.lineup_score, reverse=False))

    min_cost_def = functions.lineup_cost(min(gkp_combinations, key=functions.lineup_cost))
    min_cost_mid = functions.lineup_cost(min(mid_combinations, key=functions.lineup_cost))
    min_cost_fwd = functions.lineup_cost(min(fwd_combinations, key=functions.lineup_cost))

    max_cost_def = functions.lineup_cost(max(gkp_combinations, key=functions.lineup_cost))
    max_cost_mid = functions.lineup_cost(max(mid_combinations, key=functions.lineup_cost))
    max_cost_fwd = functions.lineup_cost(max(fwd_combinations, key=functions.lineup_cost))

    max_score_gkp = functions.lineup_score(max(gkp_combinations, key=functions.lineup_score))
    max_score_def = functions.lineup_score(max(def_combinations, key=functions.lineup_score))
    max_score_mid = functions.lineup_score(max(mid_combinations, key=functions.lineup_score))
    max_score_fwd = functions.lineup_score(max(fwd_combinations, key=functions.lineup_score))

    min_cost_mid_fwd = min_cost_mid + min_cost_fwd
    min_cost_def_mid_fwd = min_cost_def + min_cost_mid + min_cost_fwd

    max_cost_mid_fwd = max_cost_mid + max_cost_fwd
    max_cost_def_mid_fwd = max_cost_def + max_cost_mid + max_cost_fwd

    max_score_mid_fwd = max_score_mid + max_score_fwd
    max_score_def_mid_fwd = max_score_def + max_score_mid + max_score_fwd

    best_score = 0
    best_lineup = list()
    buget_lower = buget * .90
    score_lower = (max_score_gkp + max_score_def + max_score_mid + max_score_fwd) * .90

    print(f'{min_cost_mid=}, {min_cost_fwd=}, {min_cost_mid_fwd=}')
    print(f'{max_score_mid=}, {max_score_fwd=}, {max_score_mid_fwd=}')
    print(f'{m_gkps=}, {m_defs=}, {m_mids=}, {m_fwds=}')
    print(f'{score_lower=}')

    def lvl0(c):
        return (buget_lower - max_cost_def_mid_fwd <= functions.lineup_cost(c) <= buget - min_cost_def_mid_fwd
            and functions.lineup_score(c) + max_score_def_mid_fwd > best_score
            and functions.lineup_score(c) + max_score_def_mid_fwd > score_lower
            and constraints.team_constraint(c))

    def lvl1(c):
        return (buget_lower - max_cost_mid_fwd  <= functions.lineup_cost(c) <= buget - min_cost_mid_fwd
            and functions.lineup_score(c) + max_score_mid_fwd > best_score
            and functions.lineup_score(c) + max_score_mid_fwd > score_lower
            and constraints.team_constraint(c)
            and constraints.gkp_def_not_same_team(c))

    def lvl2(c):
        return (buget_lower - max_cost_fwd <= functions.lineup_cost(c) <= buget - min_cost_fwd
            and functions.lineup_score(c) + max_score_fwd > best_score
            and functions.lineup_score(c) + max_score_fwd > score_lower
            and constraints.team_constraint(c))

    def lvl3(c):
        return (functions.lineup_cost(c) <= buget
            and functions.lineup_score(c) > best_score
            and functions.lineup_score(c) > score_lower
            and constraints.team_constraint(c))

    total = (
        len(gkp_combinations) *
        len(def_combinations) *
        len(mid_combinations) *
        len(fwd_combinations)
    )
    step = (
        len(mid_combinations) *
        len(fwd_combinations)
    )

    with tqdm(
        total=total,
        bar_format='{percentage:3.0f}%|{bar:20}{r_bar}',
        unit_scale=True,
        unit_divisor=2**10,
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
                                        print('-' * 100)
                                        functions.sprint(g3)

    return best_lineup


def transfers(
    candidates: List[structures.Player],
    team: List[structures.Player],
    max_transfers: int,
    budget=1_000,
) -> Tuple[List[structures.Player], List[structures.Player]]:

    def _transfers(best, n_transfers):

        if n_transfers > max_transfers:
            raise InvalidLineup

        for candidate in candidates:
            for idx, tp in enumerate(team):

                if candidate.position != tp.position:
                    continue

                # Swap old and new player.
                tmp = list(team)
                tmp[idx] = candidate

                try:
                    new = _transfers(best, n_transfers=n_transfers + 1)
                except InvalidLineup:
                    continue

                n, s = functions.lineup_score(new), functions.lineup_score(best)
                print(s, n, s/n, n_transfers)
                if functions.lineup_score(new) >= functions.lineup_score(best):
                    best = list(new)

        return best

    return _transfers(team, 0)

"""
score: 4.09049, cost: 999, xP(TP): 128
GKP(n=2, score=0.50031, cost=95)
 Player(name='Schmeichel', team='LEI', position='GKP', cost=50, score=0.26354, points=9)
 Player(name='Sánchez', team='BHA', position='GKP', cost=45, score=0.23677, points=2)
DEF(n=5, score=1.28459, cost=276)
 Player(name='Shaw', team='MUN', position='DEF', cost=55, score=0.27594, points=1)
 Player(name='Alexander-Arnold', team='LIV', position='DEF', cost=75, score=0.27272, points=6)
 Player(name='Alonso', team='CHE', position='DEF', cost=56, score=0.25851, points=15)
 Player(name='Pinnock', team='BRE', position='DEF', cost=45, score=0.24075, points=11)
 Player(name='Ayling', team='LEE', position='DEF', cost=45, score=0.23667, points=6)
MID(n=5, score=1.47335, cost=407)
 Player(name='Fernandes', team='MUN', position='MID', cost=121, score=0.39071, points=20)
 Player(name='Salah', team='LIV', position='MID', cost=125, score=0.37309, points=17)
 Player(name='Benrahma', team='WHU', position='MID', cost=61, score=0.25639, points=12)
 Player(name='Dallas', team='LEE', position='MID', cost=55, score=0.23555, points=5)
 Player(name='Bissouma', team='BHA', position='MID', cost=45, score=0.21761, points=2)
FWD(n=3, score=0.83224, cost=221)
 Player(name='Antonio', team='WHU', position='FWD', cost=76, score=0.29929, points=13)
 Player(name='Ings', team='AVL', position='FWD', cost=80, score=0.28727, points=7)
 Player(name='Toney', team='BRE', position='FWD', cost=65, score=0.24568, points=2)
"""


def test():
    old = gather.team()
    pool = functions.top_n_score_by_cost_by_positions(gather.player_pool())
    new = transfers(
        pool,
        old,
        max_transfers=2,
    )

    # print('old')
    # functions.sprint(old)

    # print('new')
    # functions.sprint(new)

    print('transfers')
    functions.tprint(old, new)

    # functions.sprint(curernt)
    # print('----')
    # functions.sprint(t)


if __name__ == "__main__":
    # functions.sprint(lineup(
    #     pool=functions.top_n_score_by_cost_by_positions(gather.player_pool()),
    #     buget=1_000,
    #     base=((),('Alexander-Arnold', 'Shaw'), ('Salah',), (),),
    # ))
    test()
