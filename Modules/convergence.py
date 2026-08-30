from collections.abc import Iterator


def iterate_until_convergence(
    iterator: Iterator[float],
    x: float,
    max_iter: int,
    tolerance: float
) -> float:

    for _ in range(max_iter):
        x_next = next(iterator)

        if abs(x_next - x) < tolerance:
            return x_next

        x = x_next

    return x
