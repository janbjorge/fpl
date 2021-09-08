import functools
import itertools
import json
import pathlib
import time
import typing as T

import numpy as np
import pandas as pd


CACHE_FOLDER = pathlib.Path("./.cache/")
E = T.TypeVar("E")
NUMBER = T.Union[int, float]


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


def flatten(t: T.List[T.List[E]]) -> T.List[E]:
    return list(itertools.chain(*t))


@functools.lru_cache(maxsize=None)
def cached_pd_csv(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)


def caverge(
    samples: T.Sequence[T.Union[float, int]],
) -> T.Union[float, int]:
    # By applying this averger we pay more attion
    # to newer values than older values.
    if not samples:
        return 0
    weights = np.cos(np.linspace(0, 1, len(samples)) * np.pi / 3)
    weights /= weights.sum()
    return np.average(samples, weights=weights)


def timeit(prefix=None):
    def outer(f):

        nonlocal prefix
        if prefix is None:
            prefix = f.__name__

        @functools.wraps(f)
        def inner(*args, **kw):
            enter = time.time()
            try:
                return f(*args, **kw)
            finally:
                print(f"{prefix} - {time.time() - enter}")

        return inner

    return outer
