import numpy as np
import scipy.sparse

# 1
# Lös begynnelsevärdesproblemet
# dy/dt = t * log(1 + y**2)
# y(0) = 1.5
# beräkna y(1.7) med fel mindre än 1e-3
print("1")

def F_u1(t, y):
    return t * np.log(1 + y**2)

def euler_forwards(f, y0, T, N):

    h = T[1]/N
    n = round((T[1] - T[0])/h)
    t = np.zeros(n+1)
    y = np.zeros(n+1)

    t[0] = T[0]
    y[0] = y0

    for i in range(n):
        y[i+1] = y[i] + h * f(t[i], y[i])
        t[i+1] = t[i] + h

    return t, y

tol = 1e-4
diff = np.inf
N = 10
while diff > tol:
    _, y_h = euler_forwards(F_u1, 1.5, np.array([0, 1.7]), N)
    _, y_2h = euler_forwards(F_u1, 1.5, np.array([0, 1.7]), N*2)
    diff = np.abs(y_h[-1] - y_2h[-1])
    N *= 2
    print("N:", N, "h:",1.7/N, "diff:", diff, "yh:", y_h[-1])


print("")
print("3")
input()
#3 Lös begynnelsevärdesproblemet



def F_u3(t, y):
    return np.array(
        [
            y[0]*y[1],
            t**2 + y[0] - y[1]**2
        ]
    )

def euler_forwards_system(f, y0, T, N):
    h = T[1]/N
    n = round((T[1] - T[0])/h)

    t = np.zeros(n+1)
    y = np.zeros((n+1, 2))

    t[0] = T[0]
    y[0] = y0

    for i in range(n):
        y[i+1] = y[i] + h * f(t[i], y[i])
        t[i+1] = t[i] + h

    return t, y

tol = 1e-4
diff = np.inf
N = 10
while diff > tol:
    _, y_h = euler_forwards_system(F_u3, np.array([1, 0]), np.array([0, 0.8]), N)
    _, y_2h = euler_forwards_system(F_u3, np.array([1, 0]), np.array([0, 0.8]), N*2)
    diff = np.abs(y_h[-1][0] - y_2h[-1][0])
    N *= 2
    print("N:", N, "h:",1.7/N, "diff:", diff, "yh:", y_h[-1][0])


print("")
print("5")
input()

def q(x):
    return 10 * (np.sin(np.pi * x))**2

def diskretering(N, q, k, Ta, Tb):
    L = 1
    h = 1 / N 

    v1 = np.ones(N-2)
    v2 = -2*np.ones(N-1)
    v3 = np.ones(N-2)

    A = scipy.sparse.diags_array(
        [v1, v2, v3],
        offsets=[-1, 0, 1]).tocsc()

    x = np.linspace(0, L, N+1)
    b = np.ones(N-1)
    for i in range(N-1):
        b[i] = -q(x[i+1])
        if i == 0:
            b[i] -= k/(h**2) * Ta
        elif i == N-2:
            b[i] -= k/(h**2) * Tb

    return k/h**2 * A, b

k = Ta = Tb = 2
N = 10
diff = np.inf
tol = 1e-5

while diff > tol:
    A, b = diskretering(N, q, k, Ta, Tb)
    T_inner_h = scipy.sparse.linalg.spsolve(A, b)
    T_h = np.concatenate(([Ta], T_inner_h, [Tb]))

    A, b = diskretering(2*N, q, k, Ta, Tb)
    T_inner_2h = scipy.sparse.linalg.spsolve(A, b)
    T_2h = np.concatenate(([Ta], T_inner_2h, [Tb]))

    diff = np.abs(T_h[int(0.5*N)] - T_2h[int(0.5*2*N)])
    print("N:", N, "diff:", diff, "T(0.5):", T_h[int(0.5*N)])
    N *= 2