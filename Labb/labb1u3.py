import numpy as np
import matplotlib.pyplot as plt

# 3a 
print("3a")
function = lambda x: x**3 * np.exp(x)
I_exact = 6 + 2*np.exp(2)
# Trapetsregeln
# h = (b-a)/N 
# I ≈ h * ∑ (f(x_i) + f(x_i+1)))/2
# Noggranhetsordning: p = 2

def trapets(f, subinterval, interval):
    integral = 0
    h = (interval[1] - interval[0])/subinterval

    for i in range(subinterval):
        x = interval[0] + i*h
        x_next = x + h
        integral += (f(x) + f(x_next)) / 2

    return h * integral

print(trapets(function, 100, np.array([0, 2])))
print(f"e_h = {np.abs(I_exact
                      - trapets(function, 100, np.array([0, 2])))}")

# 3b
# Konvergens:
# e_h ≈ C*h**p
# p = log(e_h/e_h2) / log(2)
#
# I loglog-plot av FEL mot STEGLÄNGD h:
# lutning = p
#
# Trapets: p = 2
# Simpson: p = 4
#
# Om h halveras:
# e_h / e_(h/2) ≈ 2^p
# Trapets: 4
# Simpson: 16

print("3b")
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

# Trapetsregeln för tabellvärden
# h = antal år mellan datapunkter (=1 här)
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
# Richardsonsextrapolation
#
# Trapets:
# T_h = I + C*h^2
#
# Rihcardson tar bort h^2 felet:
# I = (4*T_h - T_2h) / 3
# Ger ordninen p = 4
I_richardson = (4*T1 - T2) / 3
print("richardson", I_richardson)


# Simpsons Regel
# 
# Vikter:
# 1 4 2 4 2 ... 4 1
# Ger ordning p = 4
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

#3f.
# f(t) = a*e^(b*(t-2014))
# ln f(t) =  ln(a) + b*(t-2014)
# Linjär ersättningsmodell
# y = ln(f)
# c0 = ln(a)
# c1 = b

# lös A.T @ A)c = A.T @ y
#
# a = exp(c0)
# b = c1

A = np.column_stack((np.ones(len(t)), t-2014))
y = np.log(f)
c = np.linalg.solve(A.T @ A, A.T @ y)

print(f"a: {np.exp(c[0])}, b: {c[1]}")

#3g
def exp_func(t):
    return np.exp(c[0]) * np.exp(c[1] * (t-2014))

f_2 = np.append(f, exp_func(2023))

print(f"\nEffekten under 2023: {exp_func(2023)}")
print("Effekten under 2023 överstinger 100kW"
      if exp_func(2023) > 100
      else "Effekten under 2023 överstinger INTE 100kW")

print(f"Ttotala mängden energi under 2014 - 2023: {trapets_data(f_2, 1)}")
print("Totala mängden energi under 2014 - 2023 överstiger 350 kWår"
      if trapets_data(f_2, 1) > 350
      else "Totala mängden energi under 2014 - 2023 överstiger INTE 350 kWår")
print("\nProjektet anses lyckat" if
      trapets_data(f_2, 1) > 350 or exp_func(2023) > 100 else
      "\nProjektet anses inte som lyckat")