from collections.abc import Callable, Iterator


def fixedpoint(f: Callable[[float], float], x0: float, max_iter: int, tolerance: int) -> Iterator[tuple[int, float]]:

    x = x0
    yield (0, x)

    for i in range(1, max_iter + 1):
        x_next = f(x)

        yield (i, x_next)
        if abs(x_next - x) < tolerance:
            return

        x = x_next