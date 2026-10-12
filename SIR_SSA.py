import numpy as np
import random
import matplotlib.pyplot as plt

beta = 0.3
gamma = 1 / 7
population = 1000
tspan = [0, 120]
Y0 = [995, 5, 0]
N = 10

k = [beta, gamma, population]

def SSA(prop, stoch, X0, tspan, coeff):
    tvec = np.zeros(1)
    tvec[0] = tspan[0]
    Xarr = np.zeros([1, len(X0)])
    Xarr[0,:] = X0
    t = tvec[0]
    X = X0
    sMat = stoch()
    
    while t < tspan[1]:
        re = prop(X, coeff)
        a0 = sum(re)
        if a0 > 1e-10:
        
            # Bestäm reaktionstid
            r1 = random.random()
            tau = - np.log(r1) / a0
            t += tau
            tvec = np.append(tvec, t)
            
            # Bestäm vilken reaktion
            r2 = random.random()
            cre = np.cumsum(re)
            cre = cre/cre[-1]
            r = 0
            while cre[r] < r2:
                r += 1
            X = X + sMat[r, :]
            Xarr = np.vstack([Xarr, X])
            
        else: # Alla reaktionshastigheter 0
            print('Simuleringen avslutad vid t=', t)
            return tvec, Xarr
            
            
    return tvec, Xarr

def stoch():
    S = np.array([
        [-1, 1, 0],
        [0, -1, 1]
    ])
    return S

def prop(Y, k):
    beta = k[0]
    gamma = k[1]
    population = k[2]

    S = Y[0]
    I = Y[1]
    R = Y[2]

    a1 = beta / population * I * S
    a2 = gamma * I

    return np.array([a1, a2])

for i in range(N):
    t, Y = SSA(prop, stoch, Y0, tspan, k)

    if i == 0:
        plt.plot(t, Y[:, 0], 'y', label = "S")
        plt.plot(t, Y[:, 1], 'r', label = "I")
        plt.plot(t, Y[:, 2], 'g', label = "R")
    else:
        plt.plot(t, Y[:, 0], 'y')
        plt.plot(t, Y[:, 1], 'r')
        plt.plot(t, Y[:, 2], 'g')


plt.xlabel("tid")
plt.ylabel("population")
plt.legend()
plt.show()





