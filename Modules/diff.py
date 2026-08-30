from collections.abc import Callable

def differentiate(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-6
) -> float:

    return (f(x + h) - f(x - h)) / (2 * h)
