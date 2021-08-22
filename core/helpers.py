import functools
import itertools
import json
import pathlib
import typing


E = typing.TypeVar("E")


def file_cache(folder: pathlib.Path):

    folder.mkdir(parents=True, exist_ok=True)

    def outer(f):
        @functools.wraps(f)
        def inner(*args, **kw):
            key = functools._make_key(args, kw, typed=False)
            cache_file = folder / f"{key}.json"
            if cache_file.exists():
                with cache_file.open("r") as fd:
                    return json.load(fd)
            rv = f(*args, *kw)
            with cache_file.open("w") as fd:
                json.dump(rv, fd)
            return rv

        return inner

    return outer


def flatten(t: typing.List[typing.List[E]]) -> typing.List[E]:
    return list(itertools.chain(*t))
