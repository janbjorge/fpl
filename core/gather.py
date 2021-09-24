import concurrent.futures
import os
import pathlib
import shutil
import statistics
import typing as T

import numpy as np
import pandas as pd
import requests

from core import (
    helpers,
    structures,
    simulator,
)


@helpers.cache("bootstrap_static")
def bootstrap_static(url="https://fantasy.premierleague.com/api/bootstrap-static/"):
    return requests.get(url).json()


@helpers.cache("gameweek")
def gameweek(
    round: int,
    url="https://fantasy.premierleague.com/api/fixtures/",
):
    return requests.get(url, params={"event": round}).json()


def current_gameweek() -> int:
    for event in bootstrap_static()["events"]:
        if event["is_current"]:
            return event["id"] + 1
    raise ValueError


def positions():
    yield from ("GKP", "DEF", "MID", "FWD")


def team_name(team_code):
    for team in bootstrap_static()["teams"]:
        if team["code"] == team_code:
            return team["short_name"]


def team_id_team_name(team_id):
    for team in bootstrap_static()["teams"]:
        if team["id"] == team_id:
            return team["short_name"]


def position(element_type_id):
    for element_type in bootstrap_static()["element_types"]:
        if element_type["id"] == element_type_id:
            return element_type["singular_name_short"]


@helpers.cache("element_summary")
def element_summary(element_id: int):
    return requests.get(
        f"https://fantasy.premierleague.com/api/element-summary/{element_id}/"
    ).json()


def name_to_element_id(name: str) -> int:
    for element in bootstrap_static()["elements"]:
        if element["web_name"] == name:
            return element["id"]
    raise KeyError


def history(
    player: str,
    folder=pathlib.Path("data"),
) -> T.Generator[pd.Series, None, None]:

    # Data from session 2019/2020, 2020/2021 and 2021/2022
    for fold in sorted(folder.glob("*_*/"), reverse=True):

        teams = helpers.cached_csv_read(fold / "teams.csv")
        merged_gw = helpers.cached_csv_read(fold / "merged_gw.csv")
        try:
            player_gws = merged_gw.loc[merged_gw.name.str.contains(player)]
        except Exception as e:
            print(e, player)
            return
        merged = player_gws.merge(teams, left_on="opponent_team", right_on="id")
        merged.sort_values("GW", inplace=True, ascending=False)

        for _, row in merged.iterrows():
            yield row


def next_n(
    player: str,
    n: int = 3,
) -> T.Tuple[T.Tuple[int, bool], ...]:
    return tuple(
        (f["team_a"], f["is_home"]) if f["is_home"] else (f["team_h"], f["is_home"])
        for f in element_summary(name_to_element_id(player))["fixtures"][:n]
    )


def strength_next_n(
    player: str,
    n: int = 3,
) -> T.Tuple[structures.Strength, ...]:
    def _strength(team_id: int, home: bool):
        for team in bootstrap_static()["teams"]:
            if team["id"] == team_id:
                if not home:
                    return structures.Strength(
                        attack=team["strength_attack_home"],
                        defence=team["strength_defence_home"],
                        overall=team["strength_overall_home"],
                        home=home,
                    )
                return structures.Strength(
                    attack=team["strength_attack_away"],
                    defence=team["strength_defence_away"],
                    overall=team["strength_overall_away"],
                    home=home,
                )

        raise KeyError(team_id)

    return tuple(_strength(*opponent) for opponent in next_n(player=player, n=n))


def player_pool() -> T.List[structures.Player]:

    pool_pd = pd.DataFrame.from_dict(bootstrap_static()["elements"])

    pool_pd["position"] = pool_pd["element_type"].apply(position)
    pool_pd["team"] = pool_pd["team_code"].apply(team_name)

    return [
        structures.Player(
            name=row.web_name,
            team=row.team,
            position=row.position,
            cost=row.now_cost,
            points=row.total_points,
            xP=simulator.Model(row.web_name).xP(),
            news=row.news,
        )
        for _, row in pool_pd.iterrows()
    ]


@helpers.cache("my_team")
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

    picks["minutes"] = picks.element.apply(lambda r: element(r, "minutes")).apply(float)
    picks["now_cost"] = picks.element.apply(lambda r: element(r, "now_cost"))
    picks["position"] = picks.element.apply(
        lambda r: position(element(r, "element_type"))
    )
    picks["team"] = picks.element.apply(lambda r: team_name(element(r, "team_code")))
    picks["total_points"] = picks.element.apply(
        lambda r: element(r, "total_points")
    ).apply(float)
    picks["web_name"] = picks.element.apply(lambda r: element(r, "web_name"))

    return [
        structures.Player(
            name=row.web_name,
            team=row.team,
            position=row.position,
            cost=row.now_cost,
            points=row.total_points,
            xP=simulator.Model(row.web_name).xP(),
            news="",
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

    historic = (
        (
            pathlib.Path("data/2020_2021"),
            "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/gws/merged_gw.csv",
        ),
        (
            pathlib.Path("data/2020_2021"),
            "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/teams.csv",
        ),
        (
            pathlib.Path("data/2021_2022"),
            "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/gws/merged_gw.csv",
        ),
        (
            pathlib.Path("data/2021_2022"),
            "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/teams.csv",
        ),
    )

    rs = max(len(url) for *_, url in historic)

    for folder, url in historic:

        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        *_, name = url.split("/")
        file = folder / name
        print(f"{str(url):<{rs}} => {str(folder):>2}.csv")
        csv = pd.read_csv(url)
        csv.to_csv(file)

    print("Refresh - done")


def validate(
    player: str,
):

    _performed = tuple(simulator.performed(player))

    if not _performed:
        return 0

    x, *_ = np.linalg.lstsq(
        np.array([np.array(v) for _, _, v in _performed]),
        np.array([np.array(s * tp) for tp, s, _ in _performed]),
        rcond=None,
    )

    print(f"*** {player} ***")
    for tp, s, hist in simulator.performed(player):
        _xp = round(x.dot(hist) / s, 1)
        print(f"{tp:<2} - {_xp}")
