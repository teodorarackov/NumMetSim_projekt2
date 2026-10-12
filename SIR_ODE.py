import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def sir(t, y):
    S = y[0]
    I = y[1]
    R = y[2]

    beta = 0.3
    gamma = 1 / 7
    population = 1000

    Sder = - beta / population * S * I
    Ider = beta / population * S * I - gamma * I
    Rder = gamma * I

    return [Sder, Ider, Rder]

y0 = [995, 5, 0]
tspan = [0, 120]

sol = solve_ivp(sir, tspan, y0)

plt.plot(sol.t, sol.y[0], 'y', label='S')
plt.plot(sol.t, sol.y[1], 'r', label='I')
plt.plot(sol.t, sol.y[2], 'g', label='R')

plt.xlabel('Tid')
plt.ylabel('Antal personer')
plt.legend()
plt.grid()
plt.show()
    