import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt("SF1514-Projekt/data/KPI.csv", delimiter=",", skiprows=3, usecols=1)
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
plt.plot(t + 1980, y)
plt.plot(t + 1980, f_a)

print(f"R_EMS (a): {E_RMS(f_a, y)}")

plt.figure(2)
plt.plot(t + 1980, modellfel(f_a, y))


# 2b. Anpassa en linje till f(t) = d0 + d1*t + d2*sin(2*pi*t/L) + d3*cos(2*pi*t/L)
# i minsta kvadratmening, till KPI datan för 1991 - 2020 genom ställa upp och lösa
# ett linjärt ekvationsystem. då L = 8
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
plt.plot(t + 1980, y)
plt.plot(t + 1980, f_b)

print(f"R_EMS (b): {E_RMS(f_b, y)}")

plt.figure(4)
plt.plot(t + 1980, modellfel(f_b, y))

#2c anpassa L som en parameter. Genom att använda samma funktion som i 2b.
# Använd Gauss-Newtons metod för att hitta de parametrar d0, d1, d2, d3, L
# som ger bästa anpassning i minstakvadratmening,
# (använd resultatet i 2b som stargissning)

X = np.array([d[0], d[1], d[2], d[3], L]) # startgissning
N = len(t)

def f_c(x, M):
    return (M[0] + M[1]*x
            + M[2]*np.sin(2*np.pi*x/M[4])
            + M[3]*np.cos(2*np.pi*x/M[4]))

def F(f, t_i, M, y_i):
    return modellfel(f(t_i, M), y_i)

def J(X):
    return np.column_stack((
        np.ones(N),
        t,
        np.sin(2*np.pi*t/X[4]),
        np.cos(2*np.pi*t/X[4]),
        (2*np.pi*t/X[4]**2) * (
            X[3]*np.sin(2*np.pi*t/X[4]) -
            X[2]*np.cos(2*np.pi*t/X[4]))
    ))


delta = np.linalg.solve(J(X).T @ J(X), -J(X).T @ F(f_c, t, X, y))

i = 0
tol = 1e-10
max_iter = 100
while np.linalg.norm(delta) > tol and i < max_iter:
    j = J(X)
    f = F(f_c, t, X, y)
    delta = np.linalg.solve(j.T @ j, -j.T @ f)
    X += delta
    i += 1
    print(i, X, np.linalg.norm(delta))

print(f"d0 = {X[0]}\nd1 = {X[1]}\nd2 = {X[2]}\nd3 = {X[3]}\nL = {X[4]}")
print(f"R_EMS (c): {E_RMS(f_c(t, X), y)}")

plt.figure(5)
plt.plot(t + 1980, y)
plt.plot(t + 1980, f_c(t, X))

plt.figure(6)
plt.plot(t + 1980, modellfel(f_c(t, X), y))


#2d 
plt.show()