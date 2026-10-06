import numpy as np
import matplotlib.pyplot as plt

#1.1 Frammåt euler med steglängd h = 0.1
# dy/dt = f(t,y) där f(t,y) = 1 + t - y
# y(0) = 1

# Euler framåt y_n+1 = y_n + h*f(t_n, y_n)

# Begynnelsevärdesproblem:
# y = f(t, y), y(t0) = y0
# 
# Framåt Euler:
# y_(n+1)' = y_n + h*f(t_n, y_n)
# t_(n+1) = t_n + h
#
# h - steglängd: (T_slut - T_start) / N
# N = antal tidssteg
# Frammåt Euler har noggranhetsordning p = 1
# 
# globalt fel: O(h)
# lokalt fel: O(h^2)
#
# p = log(e_h / e_h(h/2)) / log(2)

def f(t, y):
    return 1 + t - y

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


def y_exact(t):
    return np.exp(-t) + t

t_num, y_num = euler_forward(f, 1, 0.1, np.array([0, 1.2]))
print("error")
error = abs(y_num[-1] - y_exact(1.2))
t = np.linspace(0, 1.2, 50)
plt.plot(t_num, y_num, label = "Euler")
plt.plot(t, y_exact(t), label = "Exakt")
plt.legend()
plt.plot()

print(np.abs(y_num[-1] - y_exact(1.2)))

#1.2
k = 5
yh_t = np.zeros(k)
errors = np.zeros(k)
accuracy = np.zeros(len(errors)-1)
for i in range(k):
    _, y_h = euler_forward(f, 1, 0.2/(2**i), np.array([0, 1.2]))
    errors[i] = np.abs(y_h[-1] - y_exact(1.2))
    yh_t[i] = y_h[-1]

print(yh_t)
print(errors)

for i in range(len(errors)-1):
    accuracy[i] = np.log(errors[i] / errors[i+1]) / np.log(2)

print(accuracy)
plt.show()