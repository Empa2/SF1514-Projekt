from collections.abc import Callable, Iterator
from diff import differentiate as diff



def newton_metod(
    f: Callable[[float], float],
    x: float,
    max_iter: int,
    tolerance: float,
    div_tolerance: float = 1e-15
) -> Iterator[tuple[int, float]]:

    for i in range(max_iter):
        derivative = diff(f, x)
        if abs(derivative) < div_tolerance:
            raise ValueError("Derivative is too close to zero")
        x_next = x - f(x) / derivative

        yield (i, x_next)

        if abs(x_next - x) < tolerance:
            return

        x = x_next