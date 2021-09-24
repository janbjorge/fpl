import functools
import json
import pathlib

import pandas as pd


CACHE_FOLDER = pathlib.Path("./.cache/")


def cache(postfix: str):
    def outer(f):

        folder = CACHE_FOLDER / postfix

        @functools.lru_cache(maxsize=None)
        def inner(*args, **kw):
            key = functools._make_key(args, kw, typed=False)
            cache = folder / f"{key}.json"
            if cache.exists():
                with cache.open("r") as fd:
                    return json.load(fd)

            rv = f(*args, *kw)

            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

            with cache.open("w") as fd:
                json.dump(rv, fd)
            return rv

        return inner

    return outer


@functools.lru_cache(maxsize=None)
def cached_csv_read(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)
