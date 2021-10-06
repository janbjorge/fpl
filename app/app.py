
from typing import (
    List,
)

from fastapi import (
    FastAPI,
    Query,
    Request,
    staticfiles,
    templating,
)

from core import (
    functions,
    gather,
    optimizers,
    structures,
)


app = FastAPI()

app.mount("/static", staticfiles.StaticFiles(directory="app/static"), name="static")
templates = templating.Jinja2Templates(directory="app/static/templates")


@app.get("/lineup")
async def _lineup(
    request: Request,
    must: List[str] = Query([]),
    buget: int = 100,
    ignore: List[str] = Query([]),
):

    pool = functions.remove_bad(
        gather.player_pool(),
        None,
        must=set(must),
    )

    lineup = optimizers.lineup(
        pool=pool,
        buget=buget * 10,
        ignore=tuple(ignore) if ignore else tuple(),
        base=(
            tuple(p.name for p in pool if p.name in must and p.position == "GKP"),
            tuple(p.name for p in pool if p.name in must and p.position == "DEF"),
            tuple(p.name for p in pool if p.name in must and p.position == "MID"),
            tuple(p.name for p in pool if p.name in must and p.position == "FWD"),
            ),
        )

    return templates.TemplateResponse("lineup.html", {"players": lineup, "request": request})


@app.get("/transfer/{team_id}/")
async def _transfer(
    request: Request,
    team_id: int,
    max_transfers: int = 2,
):

    old = gather.team(team_id)
    pool = functions.remove_bad(
        gather.player_pool(),
        None,
        must=set(),
    )

    new = optimizers.Transfer.solve(
        old=old,
        pool=pool,
        tc=structures.TransferConstraints(
            max_transfers=max_transfers,
            add=tuple(),
            remove=tuple(),
            ignore=tuple(),
            buget=functions.lineup_cost(old),
        ),
    )

    return templates.TemplateResponse("lineup.html", {"players": new, "request": request})
