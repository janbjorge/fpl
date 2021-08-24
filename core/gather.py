import concurrent.futures
import os
import shutil
import typing as T

import numpy as np
import pandas as pd
import requests

from core import (
    functions,
    helpers,
    structures,
)


class ScoreWeight:
    # Weight are applied before the values are
    # sendt to the sigmoid function.
    difficulty = 3
    minutes = 1
    total_poins = 5


@helpers.file_cache("bootstrap_static")
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


@helpers.file_cache("element_summary")
def element_summary(element_id: int):
    return requests.get(
        f"https://fantasy.premierleague.com/api/element-summary/{element_id}/"
    ).json()


def difficulty(element_id, n=5) -> float:
    # Normalized 0 -> 1, where 0 is easy and 1 is hard, the
    # upcomming match is more importent than a match in a few gameworks.
    next_n = tuple(
        1 - (e["difficulty"] / 5) for e in element_summary(element_id)["fixtures"][:n]
    )
    return functions.caverge(next_n)


def history(element_id: int):
    return element_summary(element_id)["history"]


def total_points_history(element_id: int) -> float:
    return functions.caverge(
        tuple(h["total_points"] for h in reversed(history(element_id)))
    )


def minutes_history(element_id: int) -> float:
    return functions.caverge(tuple(h["minutes"] for h in reversed(history(element_id))))


def score(df: pd.DataFrame) -> pd.Series:
    return (
        functions.sigmoid(
            functions.norm(df.total_points_history), ScoreWeight.total_poins
        )
        * functions.sigmoid(functions.norm(df.minutes_history), ScoreWeight.minutes)
        * functions.sigmoid(df.difficulty, ScoreWeight.difficulty)
    )


def player_pool(
    quantile: float = 0.25,
) -> T.List[structures.Player]:

    pool_pd = pd.DataFrame.from_dict(bootstrap_static()["elements"])

    # News is only a bad sign, like transfers or injuries.
    pool_pd = pool_pd.loc[pool_pd["news"].apply(len) == 0]

    pool_pd["position"] = pool_pd["element_type"].apply(position)
    pool_pd["team"] = pool_pd["team_code"].apply(team_name)

    # The "difficulty" function returls difficulty from 0 -> 1
    # we want players with a low difficulty to have an advantaged
    pool_pd["difficulty"] = pool_pd.id.apply(difficulty)
    pool_pd["minutes_history"] = pool_pd.id.apply(minutes_history)
    pool_pd["total_points_history"] = pool_pd.id.apply(total_points_history)

    pool_pd = pool_pd[pool_pd["minutes_history"] > 0]
    pool_pd = pool_pd[pool_pd["total_points_history"] > 0]

    # Scores players.
    pool_pd["score"] = score(pool_pd)

    # Only pick candidates that are above averge in their position.
    pool_gkp = pool_pd[pool_pd["position"] == "GKP"]
    pool_gkp = pool_gkp.loc[
        pool_gkp["score"] > np.quantile(pool_gkp["score"], quantile)
    ]

    pool_def = pool_pd[pool_pd["position"] == "DEF"]
    pool_def = pool_def.loc[
        pool_def["score"] > np.quantile(pool_def["score"], quantile)
    ]

    pool_mid = pool_pd[pool_pd["position"] == "MID"]
    pool_mid = pool_mid.loc[
        pool_mid["score"] > np.quantile(pool_mid["score"], quantile)
    ]

    pool_fwd = pool_pd[pool_pd["position"] == "FWD"]
    pool_fwd = pool_fwd.loc[
        pool_fwd["score"] > np.quantile(pool_fwd["score"], quantile)
    ]

    pool_pd = pd.concat((pool_gkp, pool_def, pool_mid, pool_fwd))

    return [
        structures.Player(
            name=row.web_name,
            team=row.team,
            position=row.position,
            cost=row.now_cost,
            score=round(row.score, 5),
            points=row.total_points,
        )
        for _, row in pool_pd.iterrows()
    ]


@helpers.file_cache("my_team")
def my_team(
    login="https://users.premierleague.com/accounts/login/",
    redirect_uri="https://fantasy.premierleague.com/a/login",
    app="plfpl-web",
) -> T.List[structures.Player]:

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
        _me = s.get("https://fantasy.premierleague.com/api/me/").json()
        return s.get(
            f"https://fantasy.premierleague.com/api/my-team/{_me['player']['entry']}/"
        ).json()


def team():
    def element(_id, key):
        for element in bootstrap_static()["elements"]:
            if element["id"] == _id:
                return element[key]

    picks = pd.DataFrame.from_dict(my_team()["picks"])

    picks["minutes"] = picks.element.apply(lambda row: element(row, "minutes")).apply(
        float
    )
    picks["now_cost"] = picks.element.apply(lambda row: element(row, "now_cost"))
    picks["position"] = picks.element.apply(
        lambda row: position(element(row, "element_type"))
    )
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
    picks["minutes_history"] = picks.element.apply(minutes_history)
    picks["total_points_history"] = picks.element.apply(total_points_history)

    # Scores players.
    picks["score"] = score(picks)

    return [
        structures.Player(
            name=row.web_name,
            team=row.team,
            position=row.position,
            cost=row.now_cost,
            score=round(row.score, 5),
            points=row.total_points,
        )
        for _, row in picks.iterrows()
    ]


def refresh():

    if helpers.CACHE_FOLDER.exists():
        print(f"Clearing cache: {helpers.CACHE_FOLDER}")
        shutil.rmtree(helpers.CACHE_FOLDER)
        print(f"Clearing cache: {helpers.CACHE_FOLDER} - done")

    # Run all the functions that do external calls.
    print(f"Refreshing: {helpers.CACHE_FOLDER}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as wp:
        wp.map(element_summary, [e["id"] for e in bootstrap_static()["elements"]])
        wp.map(my_team)
    print(f"Refreshing: {helpers.CACHE_FOLDER} - done")
