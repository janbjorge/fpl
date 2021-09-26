
import functools
import json
import pathlib

CACHE_FOLDER = pathlib.Path("./.cache/")


def file(postfix: str):
    def outer(f):

        folder = CACHE_FOLDER / postfix

        @functools.lru_cache(maxsize=None, typed=False)
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
