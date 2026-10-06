import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

# Randvärdesproblem
# k*T''(x) = -q(x), 0<x<L
#
# randvilkor
# T(0) = Ta
# T=L = Tb

# h = L / N
# x_j = j*h

# Central differens för andraderivata
# T''(x_j) = (T_(j-1) - 2*T_j + T_(j+1))/h^2
#
# Insatt i diff ekv
# (k/h^2) * (T_(j-1) - 2*T_j + T_(j+1)) = -q(x_j)

# 3.c
def q(x):
    return 50*x**3*np.log(x+1)

def diskretisering_temperatur(N, q, k, Ta, Tb):
    L = 1
    h = L / N
    
    v1 = np.ones(N-2)
    v2 = -2*np.ones(N-1)
    v3 = np.ones(N-2)
    A = scipy.sparse.diags_array(
        [v1, v2, v3],
        offsets=[-1, 0, 1])

    x = np.linspace(0, L, N+1)
    b = np.ones(N-1)
    for i in range(N-1):
        b[i] = -q(x[i+1])
        if i == 0:
            b[i] -= k/(h**2) * Ta
        elif i == N-2:
            b[i] -= k/(h**2) * Tb

    return k/h**2 * A, b

N = 4
k = 2
Ta = Tb = 2
A, b = diskretisering_temperatur(N, q, k, Ta, Tb)
print(A)
print(b)


#3.d
N = 100
A, b = diskretisering_temperatur(N, q, k, Ta, Tb)
x = np.linspace(0, 1, N+1)
T_inner = scipy.sparse.linalg.spsolve(A, b)
T = np.concatenate(([Ta], T_inner, [Tb]))

print(T[20])
plt.figure(0)
plt.plot(x, T)


#3.e Konvergensstudie
# 
# Centraldifferensen för T'' har nogrannhetsordning
# p = 2 
# e_N = |T_N(0.7) - T_2N(0.7|
# p = log(e_N / e_2N)) / log(2)


T07 = np.zeros(8)
for i in range(len(T07)):
    N = 50*2**i
    A, b = diskretisering_temperatur(N, q, k, Ta, Tb)
    T_inner = scipy.sparse.linalg.spsolve(A, b)
    T = np.concatenate(([Ta], T_inner, [Tb]))

    T07[i] = T[int(0.7*N)]

print(T07)

errors = np.zeros(len(T07)-1)
for i in range(len(errors)):
    errors[i] = np.abs(T07[i] - T07[i+1])

print(f"Skillnader: {errors}")
p = np.zeros(len(errors)-1)
for i in range(len(p)):
    p[i] = np.log(errors[i] / errors[i+1]) / np.log(2)

print(f"p: {p}")


#3.f
def get_T07(Ta, Tb):
    N = 3200
    A, b = diskretisering_temperatur(N, q, k, Ta, Tb)
    T_inner = scipy.sparse.linalg.spsolve(A, b)
    T = np.concatenate(([Ta], T_inner, [Tb]))
    
    return T[int(0.7 * N)]

v0 = get_T07(2.0, 2.0)
v1 = get_T07(2.0 + 0.2, 2)
v2 = get_T07(2.0, 2 + 0.3)

delta_T = np.abs(v1-v0) + np.abs(v2-v0)

print("v0 = ", v0)
print("v1= ", v1)
print("v2 = ", v2)
print("Osäkerhet = ", delta_T)
print(f"T(0.7) = {v0} ± {delta_T}")

plt.figure(1)
N = 3200
x = np.linspace(0, 1, N+1)

# Ostört Ta = 2, Tb = 2
A, b = diskretisering_temperatur(N, q, k, 2.0, 2.0)
T_inner = scipy.sparse.linalg.spsolve(A, b)
T0 = np.concatenate(([Ta], T_inner, [Tb]))

# Stör Ta = 2.2, Tb = 2
A, b = diskretisering_temperatur(N, q, k, 2.3, 2.0)
T_inner = scipy.sparse.linalg.spsolve(A, b)
T1 = np.concatenate(([Ta], T_inner, [Tb]))

# Stör Ta = 2, Tb = 2.3
A, b = diskretisering_temperatur(N, q, k, 2.0, 2.3)
T_inner = scipy.sparse.linalg.spsolve(A, b)
T2 = np.concatenate(([Ta], T_inner, [Tb]))

plt.plot(x, T0, label = "Ostörd")
plt.plot(x, T1, label = "Ta = 2.2")
plt.plot(x, T2, label = "Tb = 2.3")

plt.xlabel("x")
plt.ylabel("T(x)")

plt.show()