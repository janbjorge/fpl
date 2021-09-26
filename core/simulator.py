import typing as T

import numpy as np

from core import (
    gather,
    structures,
)


def performed(
    player: str,
    window: int = 3,
) -> T.Generator[T.Tuple[float, float, T.Tuple[float, ...]], None, None]:

    assert window > 0

    historical: T.List[T.Tuple[int, structures.Strength]] = []

    for row in gather.history(player):
        if row.was_home:
            s = structures.Strength(
                attack=int(row.strength_defence_home),
                defence=int(row.strength_defence_home),
                overall=int(row.strength_overall_home),
                home=bool(row.was_home),
            )
        else:
            s = structures.Strength(
                attack=int(row.strength_defence_away),
                defence=int(row.strength_defence_away),
                overall=int(row.strength_overall_away),
                home=bool(row.was_home),
            )

        historical.append((int(row.total_points), s))

    historical.reverse()
    historical = historical[-window * window :]

    while len(historical) > window:
        ctp, cs = historical.pop()
        yield (
            ctp,
            cs.mean(),
            tuple(t * s.mean() for t, s in historical[-window:]),
        )


class Model:
    def __init__(self, player: str):
        self.player = player
        self._performed: T.Tuple[
            T.Tuple[float, float, T.Tuple[float, ...]], ...
        ] = tuple()
        self._model = np.zeros(3)

    @property
    def performed(self):
        if not self._performed:
            self._performed = tuple(performed(self.player))
        return self._performed

    def train(self):

        if not self.performed:
            return

        self._model, *_ = np.linalg.lstsq(
            np.array([np.array(v) for _, _, v in self.performed]),
            np.array([np.array(s * tp) for tp, s, _ in self.performed]),
            rcond=None,
        )

    def xP(self) -> float:

        if not self._model.all():
            self.train()

        next_team_stg = gather.strength_next_n(self.player, n=1)[0]

        if not self.performed:
            return 0

        last3 = self.performed[0][-1]
        assert self._model is not None
        return round(self._model.dot(last3) / next_team_stg.mean(), 1)
