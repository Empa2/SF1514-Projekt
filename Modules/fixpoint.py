from collections.abc import Callable


def fixedpoint(
    f: Callable[[float], float],
    x0: float,
    max_iter: int,
    tolerance: float
) -> float:

    x = x0

    for _ in range(max_iter):
        x_next = f(x)

        if abs(x_next - x) < tolerance:
            return x_next

        x = x_next

    return x
