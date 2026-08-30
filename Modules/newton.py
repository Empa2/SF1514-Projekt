from collections.abc import Callable, Iterator
from diff import differentiate
from convergence import iterate_until_convergence


def newton_iterator(
    f: Callable[[float], float],
    x0: float,
    div_tolerance: float = 1e-15
) -> Iterator[float]:

    x = x0
    yield x

    while True:
        derivative = differentiate(f, x)

        if abs(derivative) < div_tolerance:
            raise ValueError("Derivative is too close to zero")

        x = x - f(x) / derivative
        yield x


def newton(
    f: Callable[[float], float],
    x0: float,
    max_iter: int,
    tolerance: float
) -> float:

    iterator = newton_iterator(f, x0)
    x = next(iterator)

    return iterate_until_convergence(
        iterator, x, max_iter, tolerance
    )
