
import pathlib
import typing as T

from matplotlib import cm
from tensorflow import keras
from tensorflow import random as tf_random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import (
    structures,
    helpers,
)

SEED = 1337
tf_random.set_seed(SEED)


def model():

    model = keras.Sequential()

    model.add(keras.layers.Dense(
        10,
        input_dim=3,
    ))
    model.add(keras.layers.LeakyReLU())

    # model.add(keras.layers.Dense(
    #     10,
    #     # kernel_regularizer=keras.regularizers.l2(1e-2),
    # ))
    # model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Dense(
        1,
        # kernel_regularizer=keras.regularizers.l2(1e-2),
    ))

    model.compile(
        loss='mse',
        optimizer=keras.optimizers.SGD(learning_rate=0.05, momentum=1e-2),
    )
    return model


def exercises(
    player: str,
    folder=pathlib.Path("data"),
):

    def me_strength(teams: pd.DataFrame, name: str) -> structures.TeamStrength:
        team = teams[teams["name"] == name]
        return structures.TeamStrength(
            name=name,
            attack=int(team.strength_attack_home + team.strength_attack_away) // 2,
            defence=int(team.strength_defence_home + team.strength_defence_away) // 2,
            overall=int(team.strength_overall_home + team.strength_overall_away) // 2,
        )

    def opponent_strength(teams: pd.DataFrame, _id: int) -> structures.TeamStrength:
        team = teams[teams["id"] == _id]
        return structures.TeamStrength(
            name=team.name.values[0],
            attack=int(team.strength_attack_home + team.strength_attack_away) // 2,
            defence=int(team.strength_defence_home + team.strength_defence_away) // 2,
            overall=int(team.strength_overall_home + team.strength_overall_away) // 2,
        )

    for fold in sorted(folder.glob("*_*/"), reverse=True):

        teams = helpers.cached_pd_csv(fold / "teams.csv")
        merged_gw = helpers.cached_pd_csv(fold / "merged_gw.csv")
        player_gws = merged_gw[merged_gw.name.str.contains(player)]
        player_gws = player_gws.sort_values("GW", ascending=True)

        for _, row in player_gws.iterrows():
            mt = me_strength(teams, row.team)
            ot = opponent_strength(teams, row.opponent_team)
            yield structures.TrainingObservation(
                gw=row.GW,
                player=structures.TPlayer(
                    name=player,
                    position=row.position,
                    points_gained=row.total_points
                ),
                # TODO: Does this make sense?
                # we good, they bad? -> attach -> high
                #                    -> def    -> high
                RStrength=structures.RStrength(
                    attack=round((mt.attack - ot.defence) / (mt.attack + ot.defence), 2),
                    defence=round((mt.defence - ot.attack) / (mt.defence + ot.attack), 2),
                    overall=round((mt.overall - ot.overall) / (mt.overall + ot.overall), 2),
                ),
            )


def norm(v):
    v_max = np.max(v, axis=0)
    v_min = np.min(v, axis=0)
    return (v - v_min) / (v_max - v_min)


if __name__ == "__main__":
    import sys
    samples = tuple(exercises(sys.argv[1]))

    for sprint in samples:
        print(sprint)

    N = 5_000
    n = max(1, round(N / len(samples)) if len(samples) < N else 1)

    print(f'{n=}')

    y = np.array([s.player.points_gained for s in samples] * n)
    x = np.array([np.array((s.RStrength.attack, s.RStrength.defence, s.RStrength.overall)) for s in samples] * n)
    x = norm(x)
    print(len(x), len(y))
    assert len(x) ==  len(y)
    m = model()
    m.build()
    m.fit(
        x=x,
        y=y,
        epochs=1_000,
        verbose=1,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping("loss", mode="min", patience=4),
            keras.callbacks.ReduceLROnPlateau("loss", patience=2, factor=0.5),
        ],
    )

    for i in x[:20]:
        print(i, m.predict(np.array([i])))

    xx1, xx2 = np.meshgrid(
        np.linspace(-0.05, 1.05, num=50),
        np.linspace(-0.05, 1.05, num=50),
    )

    yy = m.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    yy = yy.reshape(xx1.shape)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter([v for v, _ in x], [v for _, v in x], y)
    surf = ax.plot_surface(xx1, xx2, yy, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # plt.contourf(xx1, xx2, yy, 10)
    # plt.plot([v for v, _ in x], [v for _, v in x], 'r*')
    # plt.colorbar()
    plt.title(sys.argv[1])
    plt.xlabel("RStrength.attack")
    plt.ylabel("RStrength.defence")
    plt.grid(True)
    plt.show()
