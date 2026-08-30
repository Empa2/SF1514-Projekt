from collections.abc import Callable, Iterator
from convergence import iterate_until_convergence


def sekant_iterator(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    div_tolerance: float = 1e-15
) -> Iterator[float]:

    x_prev = x0
    x = x1

    yield x_prev
    yield x

    while True:
        denominator = f(x) - f(x_prev)

        if abs(denominator) < div_tolerance:
            raise ValueError("Denominator is too close to zero")

        x_next = x - f(x) * (x - x_prev) / denominator

        x_prev = x
        x = x_next

        yield x

def sekant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    max_iter: int,
    tolerance: float
) -> float:

    iterator = sekant_iterator(f, x0, x1)

    next(iterator)
    x = next(iterator)

    return iterate_until_convergence(
        iterator, x, max_iter, tolerance
    )
