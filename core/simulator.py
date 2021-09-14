import typing as T

import numpy as np
import pandas as pd

from core import (
    errors,
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

        tp = int(row.total_points)

        historical.append((tp, s))

    historical.reverse()
    historical = historical[-window * window :]

    while len(historical) > window:
        ctp, cs = historical.pop()
        yield (
            ctp,
            cs.mean(),
            tuple(t * s.mean() for t, s in historical[-window:]),
        )


def lstsq_xP(
    player: str,
) -> int:

    _performed = tuple(performed(player))

    if not _performed:
        return 0

    x, *_ = np.linalg.lstsq(
        np.array([np.array(v) for _, _, v in _performed]),
        np.array([np.array(s * tp) for tp, s, _ in _performed]),
        rcond=None,
    )

    next_team_stg = gather.strength_next_n(player, n=1)[0]
    last3 = _performed[0][-1]

    return round(x.dot(last3) / next_team_stg.mean(), 1)


class Palantir:
    def __init__(self, player: str):
        self.player = player
        self._x: T.List[np.ndarray] = []
        self._y: T.List[np.ndarray] = []
        self._model: T.Optional[np.ndarray] = None

    def observation(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x.append(x)
        self._y.append(y)

    def performed(self) -> None:
        for tp, s, v in performed(self.player):
            self.observation(
                x=np.array(v),
                y=np.array(s * tp),
            )

    def fit(self) -> None:
        self._model, *_ = np.linalg.lstsq(
            np.array(self._x),
            np.array(self._y),
            rcond=None,
        )

    def predict(self, next_team_stg: structures.Strength) -> float:
        if self._model is None:
            raise errors.NotTrained(
                "You need to call `.fit(..)` before `predict(...)` can be used."
            )
        assert self._model is not None

        return round(
            self._model.dot(self._x[-1]) / next_team_stg.mean(),
            1,
        )
