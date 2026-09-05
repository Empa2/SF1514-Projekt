import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt("SF1514-Projekt/data/KPI.csv", delimiter=",", skiprows=3, usecols=1)
t = np.arange(y.size) / 12
plt.figure(0)
plt.plot(t + 1980, y)

I = (t+1980 >= 1991) & (t+1980 < 2021)
t = t[I]
y = y[I]

# 2a. Anpassa en linje till f(t) = c0 + c1*t i minsta kvadratmening, till KPI datan
# för 1991 - 2020 genom ställa upp och lösa ett linjärt ekvationsystem

# Minstakvadrat mening: (A^T * A)c = A^T * y
A = np.column_stack((np.ones(len(t)), t))
c = np.linalg.solve(A.T @ A, A.T @ y)
f = A @ c


plt.show()