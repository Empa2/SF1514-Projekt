import numpy as np
import matplotlib.pyplot as plt

# 1)
# Newton metod för att bestämma f(x), intervall -2 <= x <= 0
# Tolerance 1e-8

def f(x):
    return np.sin(x) + x**2 - 1

def df(x):
    return np.cos(x) + 2*x

def newton(xf, xdf, x0, tol, mi):
    for _ in range(mi):
        x = x0 - xf(x0) / xdf(x0)
        diff = np.abs(x-x0)
        if diff < tol:
            return x
        x0 = x
    print("Max iterations uppnått")

print(f"1: {newton(f, df, -1, 1e-8, 100)}")

# Kolla |g'(x*) < 1|
# Vid x* = 1
input("Fortsätt till uppgift 2: [Enter]: ")

def dg(x, a):
    return 2*x + a

a = np.array([-3.5, -2.5, -0.5, 0, 0.5, 2.5, 3.5])
for i, k in enumerate(a):
    print(i, k,
          np.abs(dg(1, k)),end=" ")
    print("konvergens" if np.abs(dg(1, k)) < 1
          else "divergense")

input("Fortsätt till uppgift 3: [Enter]: ")

t = np.array((0, 2, 4, 6, 8))
y = np.array((0, 1.5, 5, 7.9, 12))

A = np.column_stack((
    np.ones(len(t)),
    t,
    t**2
))
c = np.linalg.solve(A.T @ A, A.T @ y)
f3 = A @ c

print(np.sqrt(1 / len(f3) * np.sum((f3 - y)**2)))

input("Fortsätt till uppgift 5: [Enter]: ")

t5 = np.array([
    2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022
])
y5 = np.array([
    12.00, 15.10, 19.01, 23.92, 30.11,
    37.90, 47.70, 60.03, 75.56
])
B = np.column_stack((
    np.ones(len(t5)),
    t5-2014
))
y5ln = np.log(y5 / (t5-2000))
d = np.linalg.solve(B.T @ B, B.T @ y5ln)
print(np.exp(d[0]), d[1])
ft = lambda t: np.exp(d[0])*(t-2000) * np.exp(d[1] * (t-2014))
print(ft(2023))