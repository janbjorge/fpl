import functools
import itertools
import json
import pathlib
import typing


CACHE_FOLDER = pathlib.Path("./.file_cache/")
E = typing.TypeVar("E")


def file_cache(postfix: str):
    def outer(f):

        folder = CACHE_FOLDER / postfix

        @functools.wraps(f)
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


def flatten(t: typing.List[typing.List[E]]) -> typing.List[E]:
    return list(itertools.chain(*t))
