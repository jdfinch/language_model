
import itertools as it

import typing as T


E = T.TypeVar('E')

def peek(iterable: T.Iterable[E]) -> tuple[E | None, T.Iterable[E]]:
    iterating = iter(iterable)
    try:
        e = next(iterating)
        i = it.chain((e,), iterating)
    except StopIteration:
        e = None
        i = iterable
    return e, i

