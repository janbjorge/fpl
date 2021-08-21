from statistics import mean
from typing import List
import functools
import os

import pandas as pd
import requests

from core import (
    functions,
    helpers,
    structures,
)


class ScoreWeight:
    total_poins = 1
    minutes = 0.5
    selected_by_percent = 0.5
    difficulty = 0.75


@functools.lru_cache()
def bootstrap_static(url="https://fantasy.premierleague.com/api/bootstrap-static/"):
    return requests.get(url).json()


def positions():
    yield from ("GKP", "DEF", "MID", "FWD")


def team_name(team_code):
    for team in bootstrap_static()["teams"]:
        if team["code"] == team_code:
            return team["short_name"]


def position(element_type_id):
    for element_type in bootstrap_static()["element_types"]:
        if element_type["id"] == element_type_id:
            return element_type["singular_name_short"]


@helpers.file_cache()
def element_summary(element_id: int):
    return requests.get(
        f"https://fantasy.premierleague.com/api/element-summary/{element_id}/"
    ).json()


def difficulty(element_id, n=5) -> float:
    # Normalized 0 -> 1, where 0 is easy and 1 is hard.
    return sum(e["difficulty"] for e in element_summary(element_id)["fixtures"][:n]) / (
        n * 5
    )


def player_pool(
    acc=mean,
) -> List[structures.Player]:

    pool_pd = pd.DataFrame.from_dict(bootstrap_static()["elements"])

    # News is only a bad sign, like transfers or injuries.
    pool_pd = pool_pd.loc[pool_pd["news"].apply(len) == 0]

    pool_pd["position"] = pool_pd["element_type"].apply(position)
    pool_pd["team"] = pool_pd["team_code"].apply(team_name)
    pool_pd["selected_by_percent"] = pool_pd["selected_by_percent"].apply(float)

    # The "difficulty" function returls difficulty from 0 -> 1
    # we want players with a low difficulty to have an advantaged
    pool_pd["difficulty"] = 1 - pool_pd.id.apply(difficulty)

    pool_pd["score"] = (
        functions.sigmoid(
            functions.norm(pool_pd.total_points) * ScoreWeight.total_poins
        )
        * functions.sigmoid(functions.norm(pool_pd.minutes) * ScoreWeight.minutes)
        * functions.sigmoid(
            functions.norm(pool_pd.selected_by_percent)
            * ScoreWeight.selected_by_percent
        )
        * functions.sigmoid(functions.norm(pool_pd.difficulty) * ScoreWeight.difficulty)
    )

    # Only pick candidates that are above averge in their position.
    pool_gkp = pool_pd[pool_pd["position"] == "GKP"]
    pool_gkp = pool_gkp.loc[pool_gkp["score"] > acc(pool_gkp["score"])]

    pool_def = pool_pd[pool_pd["position"] == "DEF"]
    pool_def = pool_def.loc[pool_def["score"] > acc(pool_def["score"])]

    pool_mid = pool_pd[pool_pd["position"] == "MID"]
    pool_mid = pool_mid.loc[pool_mid["score"] > acc(pool_mid["score"])]

    pool_fwd = pool_pd[pool_pd["position"] == "FWD"]
    pool_fwd = pool_fwd.loc[pool_fwd["score"] > acc(pool_fwd["score"])]

    pool_pd = pd.concat([pool_gkp, pool_def, pool_mid, pool_fwd])
    pool_pd["score"] = pool_pd.score.apply(lambda x: round(x, 5))

    return [
        structures.Player(
            name=row.web_name,
            team=row.team,
            position=row.position,
            cost=row.now_cost,
            score=row.score,
            points=row.total_points,
        )
        for _, row in pool_pd.iterrows()
    ]


@functools.lru_cache()
def me(
    login="https://users.premierleague.com/accounts/login/",
    redirect_uri="https://fantasy.premierleague.com/a/login",
    app="plfpl-web",
    my_team="https://fantasy.premierleague.com/api/my-team/{}/",
) -> List[structures.Player]:

    with requests.session() as s:
        s.post(
            login,
            data={
                "password": os.environ["FPL_PASSWORD"],
                "login": os.environ["FPL_EMAIL"],
                "redirect_uri": redirect_uri,
                "app": app,
            },
        )
        return s.get(my_team.format(os.environ["FPL_TEAM_ID"])).json()


def team():
    def element(_id, key):
        for element in bootstrap_static()["elements"]:
            if element["id"] == _id:
                return element[key]

    picks = pd.DataFrame.from_dict(me()["picks"])

    picks["minutes"] = picks.element.apply(lambda row: element(row, "minutes")).apply(
        float
    )
    picks["now_cost"] = picks.element.apply(lambda row: element(row, "now_cost"))
    picks["position"] = picks.element.apply(
        lambda row: position(element(row, "element_type"))
    )
    picks["selected_by_percent"] = picks.element.apply(
        lambda row: element(row, "selected_by_percent")
    ).apply(float)
    picks["team"] = picks.element.apply(
        lambda row: team_name(element(row, "team_code"))
    )
    picks["total_points"] = picks.element.apply(
        lambda row: element(row, "total_points")
    ).apply(float)
    picks["web_name"] = picks.element.apply(lambda row: element(row, "web_name"))

    # The "difficulty" function returls difficulty from 0 -> 1
    # we want players with a low difficulty to have an advantaged
    picks["difficulty"] = 1 - picks.element.apply(difficulty)

    picks["score"] = (
        functions.sigmoid(functions.norm(picks.total_points) * ScoreWeight.total_poins)
        * functions.sigmoid(functions.norm(picks.minutes) * ScoreWeight.minutes)
        * functions.sigmoid(
            functions.norm(picks.selected_by_percent) * ScoreWeight.selected_by_percent
        )
        * functions.sigmoid(functions.norm(picks.difficulty) * ScoreWeight.difficulty)
    )

    picks["score"] = picks.score.apply(lambda x: round(x, 5))

    return [
        structures.Player(
            name=row.web_name,
            team=row.team,
            position=row.position,
            cost=row.now_cost,
            score=row.score,
            points=row.total_points,
        )
        for _, row in picks.iterrows()
    ]


if __name__ == "__main__":
    functions.sprint(team())
