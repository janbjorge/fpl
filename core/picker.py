import argparse

from plotille import (
    histogram,
)

from core import (
    functions,
    gather,
    optimizers,
    settings,
    structures,
)


def argument_parser():
    parser = argparse.ArgumentParser(prog="Lazy FPL")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enables verbose mode.",
    )
    parser.add_argument(
        "-r",
        "--refresh",
        action="store_true",
        help="Refreshes locally cached FPL APIs.",
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
    transfer_parser.add_argument(
        "-r",
        "--remove",
        type=str,
        nargs="+",
        default=[],
        help="Player(s) that must best be removed from lineup.",
    )
    transfer_parser.add_argument(
        "-a",
        "--add",
        type=str,
        nargs="+",
        default=[],
        help="Player(s) that must best be added to the lineup.",
    )
    transfer_parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        nargs="+",
        default=[],
        help="Lineup(s) containing these player(s) won't be considered.",
    )
    transfer_parser.add_argument(
        "-xp",
        "--expected-points",
        type=float,
        default=None,
        help="The players whos expected points are below this value will not be part of the player pool.",
    )

    lineup_parser = sub_parsers.add_parser(
        "lineup",
    )
    lineup_parser.add_argument(
        "-g",
        "--goalkeepers",
        nargs="+",
        help="Goalkeepers (FPL web-name) tha must be in the lineup.",
        default=[],
    )
    lineup_parser.add_argument(
        "-d",
        "--defenders",
        nargs="+",
        help="Defenders (FPL web-name) tha must be in the lineup.",
        default=[],
    )
    lineup_parser.add_argument(
        "-m",
        "--midfielders",
        nargs="+",
        help="Midfielders (FPL web-name) tha must be in the lineup.",
        default=[],
    )
    lineup_parser.add_argument(
        "-f",
        "--forwards",
        nargs="+",
        help="Forwards (FPL web-name) tha must be in the lineup.",
        default=[],
    )
    lineup_parser.add_argument(
        "-b",
        "--buget",
        help="The size of your buget, defualt is: 100.",
        default=1_00,
        type=float,
    )
    lineup_parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        nargs="+",
        default="",
        help="Lineup(s) containing these player(s) won't be considered.",
    )
    lineup_parser.add_argument(
        "-xp",
        "--expected-points",
        type=float,
        default=None,
        help="The players whos expected points are below this value will not be part of the player pool.",
    )

    print_parser = sub_parsers.add_parser(
        "print",
    )
    print_parser.add_argument(
        "show",
        nargs="?",
        choices=("team", "pool", "gmw"),
        help="Print current FPL-team, player pool or optimal gmw-team.",
    )

    histogram_parser = sub_parsers.add_parser("histogram")
    histogram_parser.add_argument(
        "-p",
        "--position",
        nargs="+",
        default=tuple(gather.positions()),
        choices=tuple(gather.positions()),
        help="Display histogram for a position(s).",
    )

    return parser.parse_args()


def main():
    parsed = argument_parser()

    settings.Global.verbose = parsed.verbose

    if parsed.refresh:
        gather.refresh()

    if parsed.mode == "transfer":
        old = gather.team()
        pool = functions.remove_bad(
            gather.player_pool(),
            parsed.expected_points,
            must=set(parsed.add),
        )
        new = optimizers.Transfer.solve(
            old=old,
            pool=pool,
            tc=structures.TransferConstraints(
                max_transfers=parsed.max,
                add=parsed.add,
                remove=parsed.remove,
                ignore=parsed.ignore,
                buget=functions.lineup_cost(old),
            ),
        )
        functions.tprint(old, new)

    elif parsed.mode == "lineup":
        pool = functions.remove_bad(
            gather.player_pool(),
            parsed.expected_points,
            must=set(
                parsed.goalkeepers
                + parsed.defenders
                + parsed.midfielders
                + parsed.forwards
            ),
        )
        functions.sprint(
            optimizers.lineup(
                pool=pool,
                buget=int(parsed.buget * 10),
                ignore=parsed.ignore,
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
            functions.lprint(gather.player_pool())
        elif parsed.show == "gmw":
            functions.sprint(
                optimizers.gmw_lineup(
                    gather.team(),
                )
            )

    elif parsed.mode == "histogram":
        xp = [p.xP for p in gather.player_pool() if p.position in parsed.position]
        print(
            histogram(
                xp,
                bins=20,
                height=20,
                width=80,
                X_label="xp",
                x_min=min(xp) * 1.05,
                x_max=max(xp) * 1.05,
            )
        )
        print()
        functions.summary(xp)


if __name__ == "__main__":
    main()
