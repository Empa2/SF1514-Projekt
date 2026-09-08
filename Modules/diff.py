from collections.abc import Callable
import numpy as np

def differentiate(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-6
) -> float:

    return (f(x + h) - f(x - h)) / (2 * h)

def jacobian(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float)

    n = len(x)
    m = len(f0)

    J = np.zeros((m, n))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = h

        J[:, i] = (f(x + dx) - f(x - dx)) / (2 * h)

    return J
