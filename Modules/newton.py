from collections.abc import Callable
import numpy as np
from diff import (differentiate, jacobian)


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


def gauss_newton(F, X, max_iter, tolerance):
    for i in range(max_iter):

        j = jacobian(F, X)
        residual = F(X)

        delta = np.linalg.solve(
            j.T @ j,
            -j.T @ residual
        )

        X = X + delta

        if np.linalg.norm(delta) < tolerance:
            break

    return X