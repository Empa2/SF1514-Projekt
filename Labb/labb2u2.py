import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# y'_n = F(t, y)
# y'_n+1 = y_n + h * F(t, y)

# y = [q, i]
# i(t) = dq / dt

# i' = -R/L * i - 1/(CL) * q
# q' = i
# y' = [q', i']

# 2b, skriv upp systemet y' = F(t, y) som en funktion

def F(t, y, R, L, C):
    q = y[0]
    i = y[1]
    return np.array([i, -R/L*i - 1/(C*L)*q])


# 2.c Solve_ivp används som referenslösning
#
# sol.t     = tidpunkterna
# sol.y     = lösningarna
# sol.y[0]  = q(t)
# sol.y[1]  = i(t)
#
# mindre rtol och atol ger nogrannare referenslösning

t_span = (0, 20)
y0 = np.array([1, 0])
R, L, C = 1, 2, 0.5
sol1 = scipy.integrate.solve_ivp(
    F,t_span,y0,args=(R, L, C)
)
R, L, C = 0, 2, 0.5
sol2 = scipy.integrate.solve_ivp(
    F,t_span,y0,args=(R, L, C),
)

plt.figure(1)
plt.plot(sol1.t, sol1.y[0], label="q(t)")
plt.plot(sol1.t, sol1.y[1], label="i(t)")
plt.legend()

plt.figure(2)
plt.plot(sol2.t, sol2.y[0], label="q(t)")
plt.plot(sol2.t, sol2.y[1], label="i(t)")
plt.legend()

#2d Frammåt euler, och stabilitet

def euler_forward_system(f, y0, h, T, R, L, C):
    T0 = T[0]
    T_end = T[1]
    n = round((T_end - T0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, 2))

    t[0] = T0
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i], R, L, C)
        t[i + 1] = t[i] + h

    return t, y

# Stabilitet
#
# Framåt euler är inte stabil för godtyckligt stora h
# för stort tidssteg kan ge växande / oscillerande numerisk lösning
# trots att den verkliga lösningen är stabil

# Större N -> Mindre h, bättre stabilitet

y0 = np.array([1, 0])
N_values = np.array([20, 40, 80, 160])

for N in N_values:
    h = (20 - 0) / N
    t_num, y_num = euler_forward_system(
        F,
        y0,
        h,
        np.array([0, 20]),
        1,      # R
        2,      # L
        0.5     # C
    )

    plt.figure(N)
    plt.plot(t_num, y_num[:, 0], label="q(t)")
    plt.plot(t_num, y_num[:, 1], label="i(t)")
    plt.legend()

# 2e Konvergensstudie
# Jämför Euler lösningen vid T=20, med en noggran
# referenslösning från solve_ivp

# error_q = |q_Euler(T) - q_ref(T)|
# error_i = |i_Euler(T) - i_ref(T)|

# Frammåt Euler har p = 1, så
# p = log(e_h / e_h(h/2)) / log(2)
# förväntat p = 1

R, L, C = 1, 2, 0.5
y0 = np.array([1, 0])

sol_ref = scipy.integrate.solve_ivp(
    F, [0, 20], y0,
    args = (R, L, C),
    rtol = 1e-6,
    atol = 1e-6
)

y_ref = sol_ref.y[:, -1]

N_values = np.array([80, 160, 320, 640, 1280, 2560, 5120, 10240])
errors = np.zeros((len(N_values), 2))

for j, N in enumerate(N_values):
    h = 20 / N
    t_num, y_num = euler_forward_system(
        F, y0, h, [0, 20], R, L, C
    )
    errors[j] = np.abs(y_num[-1] - y_ref)

p = np.zeros((len(N_values)-1, 2))
for j in range(len(N_values)-1):
    p[j] = np.log(errors[j] / errors[j+1]) / np.log(2)

print("Fel:")
print(errors)

print("Noggrannhetsordning:")
print(p)

plt.show()