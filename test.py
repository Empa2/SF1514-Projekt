import numpy as np
import scipy.sparse



def F(t, y):
    return t * np.log(1 + y**2)

def euler_forward(f, y0, h, T):
    T0 = T[0]
    T_end = T[1]
    n = round((T_end - T0)/h)

    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    t[0] = 0
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        t[i + 1] = t[i] + h
    return t, y


tol = 1e-4
diff = np.inf
N = 10
while diff > tol:
    h = 1.7 / N
    _, y_h = euler_forward(F, 1.5 , h, np.array([0, 1.7]))
    _, y_2h = euler_forward(F, 1.5 , 2*h, np.array([0, 1.7]))

    diff = np.abs(y_h[-1] - y_2h[-1])
    print("N =", N, "h =", h,"y =",  y_h[-1], "diff =", diff)

    N *= 2
input("")
#3
print(3)
def P(t, y):
    return np.array([
        y[0] * y[1],
        t**2 + y[0] - y[1] ** 2])

def euler_forward_system(f, y0, h, T):
    T0 = T[0]
    T_end = T[1]
    n = round((T_end - T0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, 2))

    t[0] = T0
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        t[i + 1] = t[i] + h

    return t, y

h = 0.8 / 10000
y0 = np.array([1.0, 0.0])
T = np.array([0, 0.8])
t, y = euler_forward_system(P, y0, h, T)

print(t)
print(y[-1])

input("")
#4 
def f(x, y):
    return x * y**3

x = f(1, 2)
x_a = f(1.5, 2)
x_b = f(1, 2.25)

print(np.abs(x_a-x) + np.abs(x_b -x))


#5
def q(x):
    return 10 * (np.sin(np.pi * x))**2

def diskretisering_temperatur(N, q, k, Ta, Tb):
    L = 1
    h = L / N
    
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

k = 2
Ta = Tb = 2
N = 10
A, b = diskretisering_temperatur(N, q, k, Ta, Tb)
x = np.linspace(0, 1, N+1)
T_inner = scipy.sparse.linalg.spsolve(A, b)
T = np.concatenate(([Ta], T_inner, [Tb]))


T05 = np.zeros(10)
for i in range(len(T05)):
    N = 50*2**i
    A, b = diskretisering_temperatur(N, q, k, Ta, Tb)
    T_inner = scipy.sparse.linalg.spsolve(A, b)
    T = np.concatenate(([Ta], T_inner, [Tb]))

    T05[i] = T[int(0.5*N)]

print(T05)

errors = np.zeros(len(T05)-1)
for i in range(len(errors)):
    errors[i] = np.abs(T05[i] - T05[i+1])

print(errors)