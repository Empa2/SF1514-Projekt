from collections.abc import Callable, Iterator

def sekant_metod(
    f: Callable[[float], float],
    x: float,
    x_prev: float,
    max_iter: int,
    tolerance: float,
    div_tolerance: float = 1e-15
) -> Iterator[tuple[int, float]]:

    for i in range(max_iter):
        denominator = f(x) - f(x_prev)

        if abs(denominator) < div_tolerance:
            raise ValueError("Denominator is too close to zero")
        x_next = x - f(x)*(x-x_prev)/denominator

        yield (i, x_next)

        if abs(x_next - x) < tolerance:
            return

        x_prev = x
        x = x_next