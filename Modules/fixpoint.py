from collections.abc import Callable, Iterator
from convergence import iterate_until_convergence


def fixedpoint_iterator(
    f: Callable[[float], float],
    x0: float
) -> Iterator[float]:

    x = x0
    yield x

    while True:
        x = f(x)
        yield x


def fixedpoint(
    f: Callable[[float], float],
    x0: float,
    max_iter: int,
    tolerance: float
) -> float:

    iterator = fixedpoint_iterator(f, x0)
    x = next(iterator)

    return iterate_until_convergence(
        iterator, x, max_iter, tolerance
    )
