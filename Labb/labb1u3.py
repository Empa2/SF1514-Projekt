import numpy as np
import matplotlib.pyplot as plt

# 3a 
print("3a")
function = lambda x: x**3 * np.exp(x)

def trapets(f, subinterval, interval):
    integral = 0
    h = (interval[1] - interval[0])/subinterval

    for i in range(subinterval):
        x = interval[0] + i*h
        x_next = x + h
        integral += (f(x) + f(x_next)) / 2

    return h * integral

print(trapets(function, 100, np.array([0, 2])))

# 3b
print("3b")
I_exact = 6 + 2*np.exp(2)
for j in range(10):
    N = 2**(j + 1)
    error_h = np.abs(I_exact - trapets(function, N, np.array([0, 2])))
    error_h2 = np.abs(I_exact - trapets(function, N*2, np.array([0, 2])))
    p = np.log(error_h / error_h2) / np.log(2)
    print(N, error_h, p)

#3c
print("3c")
t = np.array([
    2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022
])
f = np.array([
    12.00, 15.10, 19.01, 23.92, 30.11,
    37.90, 47.70, 60.03, 75.56
])

h = (t[-1] - t[0]) / (len(t) - 1)

def trapets_data(f, h):
    integral = 0
    for i in range(0, len(f)-h, h):
        integral += (f[i] + f[i+h]) / 2
    return integral * h


print(f"I = {trapets_data(f, 1)}")

#3d
T1 = trapets_data(f, 1)
T2 = trapets_data(f, 2)
T4 = trapets_data(f, 4)
T8 = trapets_data(f, 8)
error_1 = np.abs(T1 - T2)
error_2 = np.abs(T2 - T4)
error_4 = np.abs(T4 - T8)
p1 = np.log(error_2 / error_1) / np.log(2)
p2 = np.log(error_4 / error_2) / np.log(2)

print("e_1 = |T1 - T2|", error_1)
print("e_2 = |T2 - T4|", error_2)
print("e_4 = |T4 - T8|", error_4)
print("p1", p1)
print("p2", p2)

#3e
I_richardson = (4*T1 - T2) / 3
print("richardson", I_richardson)

def simpsons_data(f, h):
    integral = 0
    for i, _ in enumerate(f):
        if i == 0 or i == len(f)-1:
            weight = 1
        elif i % 2 == 0:
            weight = 2
        else:
            weight = 4
        integral += f[i] * weight
    return h/3 * integral

S = simpsons_data(f, 1)
print("Simpsons", S)

#3g.
# Minsta kvadrat mening för att lösa parametrarna a, b i f(t) = ae^(b*(t-2014))
# ln f(t) =  ln a + b(t-2014)
# Linjär ersättningsmodell y = ln(f(t)), c0 = ln a, c1 = b

A = np.column_stack((np.ones(len(t)), t-2014))
y = np.log(f)
c = np.linalg.solve(A.T @ A, A.T @ y)

print(f"a: {np.exp(c[0])}, b: {c[1]}")

#3f
def exp_func(t):
    return np.exp(c[0]) * np.exp(c[1] * (t-2014))

f_2 = np.append(f, exp_func(2023))
print(trapets_data(f_2, 1))