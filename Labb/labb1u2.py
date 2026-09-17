import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt("data/KPI.csv", delimiter=",", skiprows=3, usecols=1)
t = np.arange(y.size) / 12
#plt.figure(0)
#plt.plot(t + 1980, y)

I = (t+1980 >= 1991) & (t+1980 < 2021)
t = t[I]
y = y[I]


def E_RMS(f, y):
    summa = 0
    lengt = len(y)
    for i in range(lengt):
        summa += (f[i] - y[i])**2
    return np.sqrt(summa/lengt)


def modellfel(f, y):
    return f - y


# 2a. Anpassa en linje till f(t) = c0 + c1*t i minsta kvadratmening, till KPI datan
# för 1991 - 2020 genom ställa upp och lösa ett linjärt ekvationsystem

# Minstakvadrat mening: (A^T * A)c = A^T * y
A_a = np.column_stack((np.ones(len(t)), t))
c = np.linalg.solve(A_a.T @ A_a, A_a.T @ y)
f_a = A_a @ c

print(f"c0 = {c[0]}\nc1 = {c[1]}")
plt.figure(1)
plt.plot(t + 1980, y, label="KPI-data")
plt.plot(t + 1980, f_a, label="Linjär modell")
plt.xlabel("År")
plt.ylabel("KPI")
plt.legend()

print(f"R_EMS (a): {E_RMS(f_a, y)}")

plt.figure(2)
plt.plot(t + 1980, modellfel(f_a, y))
plt.title("2a: Residualer")
plt.xlabel("År")
plt.ylabel("Modell - data")

# 2b. Anpassa en linje till f(t) = d0 + d1*t + d2*sin(2*pi*t/L) + d3*cos(2*pi*t/L)
# i minsta kvadratmening, till KPI datan för 1991 - 2020 genom ställa upp och lösa
# ett linjärt ekvationsystem. då L = 8

# Linjär minstakvadrat
#  1 kolumn i A per parameter/basfunktion
# lös (A.T @ A)c = A.T @ y
 
L = 8
A_b = np.column_stack((
    np.ones(len(t)),
    t,
    np.sin(2*np.pi*t/L),
    np.cos(2*np.pi*t/L)))

d = np.linalg.solve(A_b.T @ A_b, A_b.T @ y)
f_b = A_b @ d

print(f"d0 = {d[0]}\nd1 = {d[1]}\nd2 = {d[2]}\nd3 = {d[3]}")

plt.figure(3)
plt.plot(t + 1980, y, label="KPI-data")
plt.plot(t + 1980, f_b, label="Sinusmodell (L=8)")
plt.title("2b: Linjär minstakvadrat med sinus och cosinus")
plt.xlabel("År")
plt.ylabel("KPI")
plt.legend()

print(f"R_EMS (b): {E_RMS(f_b, y)}")

plt.figure(4)
plt.plot(t + 1980, modellfel(f_b, y))
plt.title("2b: Residualer")
plt.xlabel("År")
plt.ylabel("Modell - data")

#2c anpassa L som en parameter. Genom att använda samma funktion som i 2b.
# Använd Gauss-Newtons metod för att hitta de parametrar d0, d1, d2, d3, L
# som ger bästa anpassning i minstakvadratmening,
# (använd resultatet i 2b som stargissning)

# GAUSS-NEWTON:
# F(X) = modell(X) - data
# J = Jacobian för F
# lös (J.T @ J)delta = -J.T @ F
# uppdatera X_n = X + delta
# räkna om J och F varje iteration


cGN = np.array([d[0], d[1], d[2], d[3], L]) # startgissning

def fGN(x, D):
    return (
        D[0]
        + D[1]*x
        + D[2]*np.sin(2*np.pi*x/D[4])
        + D[3]*np.cos(2*np.pi*x/D[4]))

def F(X):
    return modellfel(fGN(t, X), y)

def J(X):
    return np.column_stack((
    np.ones(len(t)),
    t,
    np.sin(2*np.pi*t/X[4]),
    np.cos(2*np.pi*t/X[4]),
    (2*np.pi*t/X[4]**2) * (
        X[3]*np.sin(2*np.pi*t/X[4]) -
        X[2]*np.cos(2*np.pi*t/X[4]))
))

i = 0
tol = 1e-10
max_iter = 1000
delta = np.ones(cGN.shape)

while np.linalg.norm(delta) > tol and i < max_iter:
    JGN = J(cGN)
    FGN = F(cGN)

    delta = np.linalg.solve(JGN.T @ JGN, -JGN.T @ FGN)
    cGN += delta
    i += 1
    print(i, cGN, np.linalg.norm(delta))

print(f"c0 = {cGN[0]}\nc1 = {cGN[1]}\nc2 = {cGN[2]}\nc3 = {cGN[3]}\nL = {cGN[4]}")
print(f"R_EMS (c): {E_RMS(fGN(t, cGN), y)}")


plt.figure(5)
plt.plot(t + 1980, y, label="KPI-data")
plt.plot(t + 1980, fGN(t, cGN), label="Gauss-Newton")
plt.title("2c: Icke-Linjär minstakvadratanpassning")
plt.xlabel("År")
plt.ylabel("KPI")
plt.legend()

plt.figure(6)
plt.plot(t + 1980, modellfel(fGN(t, cGN), y))
plt.title("2c: Residualer")
plt.xlabel("År")
plt.ylabel("Modell - data")

#2d
print("2d")
print("RMS a:", E_RMS(f_a, y))
print("RMS b:", E_RMS(f_b, y))
print("RMS c:", E_RMS(fGN(t, cGN), y))

print("Genomsnittlig KPI-ökning per år:")
print("Modell a:", (f_a[-1] - f_a[0]) / (t[-1] - t[0]))
print("Modell b:", (f_b[-1] - f_b[0]) / (t[-1] - t[0]))
print("Modell c:", (fGN(t[-1], cGN) - fGN(t[0], cGN)) / (t[-1] - t[0]))

plt.show()