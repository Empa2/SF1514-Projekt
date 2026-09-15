import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) + x**2 - 1

def df(x):
    return np.cos(x) + 2*x


def newton_metod(func, derivata, x, tolerance, max_iteration, should_print = True):
    i = 1
    diff = []
    x_next = x - func(x)/derivata(x)
    diff.append(np.abs(x_next - x))
    if should_print:
        print(i, x_next, x, diff[-1])

    while np.abs(x_next - x) >= tolerance and i < max_iteration:
        x = x_next
        x_next = x - func(x)/derivata(x)
        diff.append(np.abs(x_next - x))
        if should_print:
            print(i+1, x_next, x, diff[-1])
        i += 1

    return x_next, diff

n = newton_metod(f, df, -1, 1e-8, 100)[0]
print(n)



########

t = np.array((0, 2, 4, 6, 8))
y = np.array((0, 1.5, 5, 7.9, 12))


A = np.column_stack((np.ones(len(t)), t, t**2))
print(A)
# (A^T * A)c = A^T * y
c = np.linalg.solve(A.T @ A, A.T @ y)
f = A @ c

print(c)

# plt.plot(t,y)
# plt.show()

def E_RMS(f, y):
    summa = 0
    lengt = len(y)
    for i in range(lengt):
        summa += (f[i] - y[i])**2
    return np.sqrt(summa/lengt)
print("E_RMS")
print(E_RMS(f, y))

# Minsta kvadrat mening för att lösa parametrarna a, b i f(t) = ae^(b*(t-2014))
# ln f(t) =  ln a + b(t-2014)
# Linjär ersättningsmodell y = ln(f(t)), c0 = ln a, c1 = b

tl = np.array([
    2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022
])

fl = np.array([
    12.00, 15.10, 19.01, 23.92, 30.11,
    37.90, 47.70, 60.03, 75.56
])

A = np.column_stack((np.ones(len(tl)), tl-2014))
yl = np.log(fl / (tl-2000))
c = np.linalg.solve(A.T @ A, A.T @ yl)
print(np.exp(c[0]), c[1])

def fxLA(t, c):
    return np.exp(c[0])*(t-2000) * np.exp(c[1]*(t-2014))

print(fxLA(2023, c))