import numpy as np
import random
import matplotlib.pyplot as plt

beta = 0.3
gamma = 1 / 7
population = 1000
alfa = 0.5 #väldigt liten ger inge möten mellan folk - tar längre tid för alla andra reaktioner att ske, högre värde - tidigare sker reaktioner
# inkubationstid = 1 / alfa
mu = 0.02
v = 1

tspan = [0, 120]
Y0 = [995, 0, 5, 0, 0, 0] 
N = 4

k = [beta, gamma, population, alfa, mu, v]

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
        [-1, 1, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0],
        [0, 0, -1, 1, 0, 0],
        [0, 0, -1, 0, 1, 0],
        [-1, 0, 0, 0, 0, 1]
    ])
    return S

def prop(Y, k):
    beta = k[0]
    gamma = k[1]
    population = k[2]
    alfa = k[3]
    mu = k[4]
    v = k[5]

    S = Y[0]
    E = Y[1]
    I = Y[2]
    R = Y[3]
    D = Y[4]
    V = Y[5]

    a1 = beta / population * I * S
    a2 = alfa * E
    a3 = gamma * I
    a4 = mu * I
    a5 = v


    return np.array([a1, a2, a3, a4, a5])

for i in range(N):
    t, Y = SSA(prop, stoch, Y0, tspan, k)

    if i == 0:
        plt.plot(t, Y[:, 0], 'y', label = "S")
        plt.plot(t, Y[:, 1], color = 'orange', label = "E")
        plt.plot(t, Y[:, 2], 'r', label = "I")
        plt.plot(t, Y[:, 3], 'g', label = "R")
        plt.plot(t, Y[:, 4], color = 'black', label = "D")
        plt.plot(t, Y[:, 5], color = 'purple', label = "V")
    else:
        plt.plot(t, Y[:, 0], 'y')
        plt.plot(t, Y[:, 1], color = 'orange')
        plt.plot(t, Y[:, 2], 'r')
        plt.plot(t, Y[:, 3], 'g')
        plt.plot(t, Y[:, 4], color = 'black')
        plt.plot(t, Y[:, 5], color = 'purple')


plt.xlabel("tid")
plt.ylabel("population")
plt.legend()
plt.show()





