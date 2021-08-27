import dataclasses
import functools
import random
import typing as T

import pandas as pd

from core import (
    gather,
)


@dataclasses.dataclass(frozen=True, eq=True)
class Player:
    name: str
    team: str
    position: str
    # Goals per minute
    gpm: float
    # Saves per minute
    spm: float


@dataclasses.dataclass(frozen=True, eq=True)
class Team:
    name: str
    players: T.List[Player]

    @property
    def gpm(self) -> float:
        return sum(p.gpm for p in self.players)

    @property
    def spm(self) -> float:
        return sum(p.spm for p in self.players)


def players() -> T.List[Player]:

    pool_pd = pd.DataFrame.from_dict(gather.bootstrap_static()["elements"])

    # News is only a bad sign, like transfers or injuries.
    # keep ?
    pool_pd = pool_pd.loc[pool_pd["news"].apply(len) == 0]

    pool_pd["position"] = pool_pd["element_type"].apply(gather.position)
    pool_pd["team"] = pool_pd["team_code"].apply(gather.team_name)

    # The "difficulty" function returls difficulty from 0 -> 1
    # we want players with a low difficulty to have an advantaged
    pool_pd = pool_pd[pool_pd["minutes"] > 0]

    pool_pd["gpm"] = pool_pd.goals_scored / pool_pd.minutes
    pool_pd["spm"] = pool_pd.saves / pool_pd.minutes

    return [
        Player(
            name=row.web_name,
            team=row.team,
            position=row.position,
            gpm=round(row.gpm, 5),
            spm=round(row.spm, 5),
        )
        for _, row in pool_pd.iterrows()
    ]


@functools.lru_cache(maxsize=None)
def teams() -> T.Tuple[Team, ...]:

    def _inner():
        for t in set(p.team for p in players()):
            yield Team(
                name=t,
                players=[p for p in players() if p.team == t],
            )

    return tuple(_inner())


def match(
    t1: Team,
    t2: Team,
    duration: int = 90,
):
    t1_rgpm = t1.gpm - t2.spm if t1.gpm - t2.spm > 0 else 0.001
    t2_rgpm = t2.gpm - t1.spm if t2.gpm - t1.spm > 0 else 0.001

    t1goals = sum(random.random() < t1_rgpm for _ in range(duration))
    t2goals = sum(random.random() < t2_rgpm for _ in range(duration))

    return (
        (t1.name, t1goals),
        (t2.name, t2goals),
    )


def gameweek(round: int):
    return


def all_v_all():
    import itertools
    import collections

    c = collections.Counter()

    for t1, t2 in itertools.permutations(teams(), 2):
        (hn, hg), (an, ag) = match(t1, t2)

        if hg > ag:
            c[hn] += 3
        elif hg < ag:
            c[an] += 3
        else:
            c[hn] += 1
            c[an] += 1

    for team, porints in c.most_common(len(c)):
        print(team, porints)


