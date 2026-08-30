from collections.abc import Callable

def simpsons(
    f: Callable[[float], float],
    start: float,
    stop: float,
    subinterval: int,
) -> float:

    if subinterval % 2 != 0:
        raise ValueError("Subinterval must be even")

    integral: float = 0
    h = (stop-start)/subinterval


    for i in range(subinterval+1):
        if i == 0 or i == subinterval:
            weight = 1
        elif i % 2 == 0:
            weight = 2
        else:
            weight = 4
        print(i, weight)
        x = start + i*h
        integral += f(x)*weight

    return h/3 * integral
