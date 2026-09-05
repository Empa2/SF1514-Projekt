from collections.abc import Callable
from diff import differentiate


def newton(
    f: Callable[[float], float],
    x0: float,
    max_iter: int,
    tolerance: float
) -> float:

    x = x0

    for _ in range(max_iter):
        derivative = differentiate(f, x)

        if abs(derivative) < 1e-15:
            raise ValueError("Derivative is too close to zero")

        x_next = x - f(x) / derivative

        if abs(x_next - x) < tolerance:
            return x_next

        x = x_next

    return x
