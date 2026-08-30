from collections.abc import Callable

def trapets(
    f: Callable[[float], float],
    start: float,
    stop: float,
    subinterval: int,
) -> float:

    integral: float = 0
    h = (stop-start)/subinterval


    for i in range(subinterval):
        x = start + i*h
        x_next = x + h
        integral += (f(x) + f(x_next)) / 2

    return h * integral
