import numpy as np
import matplotlib.pyplot as plt

def f(x, L=1):
    return (8/3)*(x/L) - 3*(x/L)**2 + (1/3)*(x/L)**3 - (2/3)*np.sin(np.pi*x/L)

def df(x, L=1):
    return 8/(3*L) - 6*x/L**2 + 1/L*(x/L)**2 - (2*np.pi)/(3*L)*np.cos(np.pi*x/L)
 
def g(x, L=1):
    return 3*L/8  * (3*(x/L)**2 - 1/3*(x/L)**3 + 2/3*np.sin(np.pi*x/L))

def dg(x, L=1):
    return 3*L/8 * (6*x/L**2 - 1/L*(x/L)**2 + 2*np.pi/(3*L)*np.cos(np.pi*x/L))  

def plot_function(func ,start, stop, num, label=None):
    x = np.linspace(start, stop, num)
    y = func(x)
    plt.plot(x, y, label=label)

def fixpunkt_metod(func, x, tolerance, max_iteration, should_print = True):
    i = 1
    diff = []
    x_next = func(x)
    diff.append(np.abs(x_next - x))
    if should_print:
        print(i, x_next, x, diff[-1])
    while np.abs(x_next - x) >= tolerance and i < max_iteration:
        x = x_next
        x_next = func(x)

        diff.append(np.abs(x_next - x))
        if should_print:
            print(i+1, x_next, x, diff[-1])
        i += 1

    return x_next, diff


def newton_metod(func, derivata, x, tolerance, max_iteration, should_print = True):
    i = 1
    diff = []
    x_next = x - func(x)/derivata(x)
    diff.append(np.abs(x_next - x))
    if should_print:
        print(i, x_next, x, diff[-1])

    while np.abs(x_next - x) >= tolerance and i < max_iteration:
        x = x_next
        x_next = x - func(x)/derivata(x)
        diff.append(np.abs(x_next - x))
        if should_print:
            print(i+1, x_next, x, diff[-1])
        i += 1

    return x_next, diff


# 1a. Rita upp f(x) på intervallet 0 < x < L
plt.figure(1)
plot_function(f, 0, 1, 100, "f(x)")
plot_function(df, 0, 1, 100, "df(x)")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.grid()
plt.legend()
# På grafen syns 2 nollställen, vid ca 0.3, ca 0.84


# 1b. Vilket nollställe kan fixpunkt metoden konvergera till
plt.figure(2)
plot_function(g, 0, 1, 100, "g(x)")
plot_function(dg, 0, 1, 100, "dg(x)")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.grid()
plt.legend()
# Derivatans belopp är större än 1 vid 0.3, och mindre än 1 vid 0.84
# därmed kan fixpunkts metoded konvergera vid 0.84 men inte vid 0.3


# 1c. Använd fixpunkt metoden för att hitta ett nollställe på intervall
# 0 < x < L, (L=1), Denna kan dock konvergera mot 0, vid vissa start värden
# den är inte i definitions mängden, så man får testa annat start värde
print("Fixpunkt metod")
fixpunkt_metod(g, 0.5, 1e-10, 1000)

# 1d. Använd newton metoden för att hitta ett nollställe på intervall
# 0 < x < L, (L=1), Denna kan dock konvergera mot 0 och 1, vid vissa start värden
print("Newton metod")
newton_metod(f, df, 0.7, 1e-10, 1000)

#1e. Plotta differansen mellan fixpunkt och newton, med samma start gissning
# Använd logaritimsik skala på y-axeln (plt.semilogy-lot)
x_guess = 0.7
x_fix, fix_diff = fixpunkt_metod(g, x_guess, 1e-10, 1000, False)
x_newton, newton_diff= newton_metod(f, df, x_guess, 1e-10, 1000, False)
print(f"fixpunkt konvergerar mot: {x_fix}")
print(f"Newton konvergerar mot: {x_newton}")
plt.figure(3)
plt.semilogy(fix_diff, label="fixpunkt",)
plt.semilogy(newton_diff, label="newton")
plt.ylabel("Differans")
plt.xlabel("Iterationer")
plt.grid()
plt.legend()

plt.show()
