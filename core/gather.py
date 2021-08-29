import concurrent.futures
import dateutil.parser
import os
import pathlib
import shutil
import typing as T

import pandas as pd
import requests

from core import (
    functions,
    helpers,
    structures,
)


@helpers.file_cache("bootstrap_static")
def bootstrap_static(url="https://fantasy.premierleague.com/api/bootstrap-static/"):
    return requests.get(url).json()


@helpers.file_cache("gameweek")
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


def matchups(
    round: int,
) -> T.Tuple[T.Tuple[str, str], ...]:
    def _key(row):
        return (
            dateutil.parser.parse(row["kickoff_time"]),
            row["team_h"],
            row["team_a"],
        )

    return tuple(
        (
            team_id_team_name(fixture["team_h"]),
            team_id_team_name(fixture["team_a"]),
        )
        for fixture in sorted(gameweek(round), key=_key)
    )


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


@helpers.file_cache("element_summary")
def element_summary(element_id: int):
    return requests.get(
        f"https://fantasy.premierleague.com/api/element-summary/{element_id}/"
    ).json()


def name_to_element_id(name: str) -> int:
    for element in bootstrap_static()["elements"]:
        if element["web_name"] == name:
            return element["id"]
    raise KeyError


def difficulty_next_n(element_id, n=5) -> T.List[float]:
    # Normalized 0 -> 1, where 0 is easy and 1 is hard, the
    # upcomming match is more importent than a match in a few gameworks.
    return list((e["difficulty"]) for e in element_summary(element_id)["fixtures"][:n])


def player_pool(
    cutoff: T.Tuple[int, int, int, int] = (2, 3, 3, 2),
) -> T.List[structures.Player]:

    pool_pd = pd.DataFrame.from_dict(bootstrap_static()["elements"])

    # News is only a bad sign, like transfers or injuries.
    pool_pd = pool_pd.loc[pool_pd["news"].apply(len) == 0]

    pool_pd["position"] = pool_pd["element_type"].apply(position)
    pool_pd["team"] = pool_pd["team_code"].apply(team_name)

    return [
        p
        for p in functions.top_n_xp_by_cost_by_positions(
            [
                structures.Player(
                    name=row.web_name,
                    team=row.team,
                    position=row.position,
                    cost=row.now_cost,
                    points=row.total_points,
                    xP=functions.xP(
                        tp=row.total_points,
                        opponents=opponent_strength(row.web_name),
                        gmw=current_gameweek(),
                    ),
                )
                for _, row in pool_pd.iterrows()
            ],
            cutoff=cutoff,
        )
        if p.xP > 0
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

    return [
        structures.Player(
            name=row.web_name,
            team=row.team,
            position=row.position,
            cost=row.now_cost,
            points=row.total_points,
            xP=functions.xP(
                tp=row.total_points,
                opponents=opponent_strength(row.web_name),
                gmw=current_gameweek(),
            ),
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
            pathlib.Path("data/2019_2020"),
            "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/gws/merged_gw.csv",
        ),
        (
            pathlib.Path("data/2019_2020"),
            "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/teams.csv",
        ),
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
        print(f"{str(url):<{rs}} => {str(folder):>2}")
        csv = pd.read_csv(url)
        csv.to_csv(file)

    print("Refresh - done")


def opponent_strength(
    player: str,
    n: int = 1,
    folder=pathlib.Path("data"),
) -> structures.Samples:

    # Obs. remember that the "most" significant sample
    # should be at `0`, thus newer samples should be at an
    # lower index than "old" samples.
    historical: T.List[T.Union[int, float]] = []

    # Data from session 2019/2020, 2020/2021 and 2021/2022
    for fold in sorted(folder.glob("*_*/"), reverse=True):

        teams = helpers.cached_pd_csv(fold / "teams.csv")
        merged_gw = helpers.cached_pd_csv(fold / "merged_gw.csv")
        player_gws = merged_gw.loc[merged_gw.name.str.contains(player)]

        merged = player_gws.merge(teams, left_on="opponent_team", right_on="id")
        merged.sort_values("GW", inplace=True, ascending=False)
        historical.extend(merged.strength.values)

    return structures.Samples(
        historical=historical,
        future=difficulty_next_n(name_to_element_id(player), n=n),
    )
