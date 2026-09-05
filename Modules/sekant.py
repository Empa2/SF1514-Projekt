from collections.abc import Callable


def sekant(
    f: Callable[[float], float],
    x0: float,
    max_iter: int,
    tolerance: float
) -> float:

    x_prev = x0
    x = x0 + 1e-5

    for _ in range(max_iter):
        denominator = f(x) - f(x_prev)

        if abs(denominator) < 1e-15:
            raise ValueError("Denominator is too close to zero")

        x_next = x - f(x) * (x - x_prev) / denominator

        if abs(x_next - x) < tolerance:
            return x_next

        x_prev = x
        x = x_next

    return x
