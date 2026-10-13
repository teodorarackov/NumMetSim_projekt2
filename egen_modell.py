import numpy as np
import random
import matplotlib.pyplot as plt

beta = 0.3 
gamma = 1 / 7 #sjuk 7 dagar/individ
population = 1000
alfa = 1 / 2 #2 dagar inkubationstid 
mu = 1 / 50 #dagar/individ innan död
lam = 5 #5 pers vacc/tidsenhet
theta = 1 / 21 #två veckors mellanrumm mellan dos1 och 2
epsilon1 = 0.5 #takt från dos1 till exposed
epsilon2 = 0.1 #takt från dos2 till exposed
phi = 1 / 14 # dagar till immun från dos2

tspan = [0, 120]
Y0 = [995, 0, 5, 0, 0, 0, 0, 0] #S, E, I, R, D, V1, V2, Im
N = 5

k = [beta, gamma, population, alfa, mu, lam, theta, epsilon1, epsilon2, phi]

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
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, -1, 0, 1, 0, 0, 0],
        [-1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0],
        [0, 1, 0, 0, 0, -1, 0, 0],
        [0, 1, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, -1, 1]
    ])
    return S

def prop(Y, k):
    beta = k[0]
    gamma = k[1]
    population = k[2]
    alfa = k[3]
    mu = k[4]
    lam = k[5]
    theta = k[6]
    epsilon1 = k[7]
    epsilon2 = k[8]
    phi = k[9]

    S = Y[0]
    E = Y[1]
    I = Y[2]
    R = Y[3]
    D = Y[4]
    V1 = Y[5]
    V2 = Y[6]
    Im = Y[7]


    a1 = beta / population * I * S
    a2 = alfa * E
    a3 = gamma * I
    a4 = mu * I
    a5 = lam  if S > 0 else 0
    a6 = theta * V1
    a7 = epsilon1 * beta / population * I * V1
    a8 = epsilon2 * beta / population * I * V2
    a9 = phi * V2


    return np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9])

for i in range(N):
    t, Y = SSA(prop, stoch, Y0, tspan, k)

    if i == 0:
        plt.plot(t, Y[:, 0], 'yellow', label = "S")
        plt.plot(t, Y[:, 1], color = 'orange', label = "E")
        plt.plot(t, Y[:, 2], 'red', label = "I")
        plt.plot(t, Y[:, 3], 'green', label = "R")
        plt.plot(t, Y[:, 4], color = 'black', label = "D")
        plt.plot(t, Y[:, 5], color = "#47144873", label = "V1")
        plt.plot(t, Y[:, 6], color = "#DA80D169", label = "V2")
        plt.plot(t, Y[:, 7], color = 'blue', label = "Im")
    else:
        plt.plot(t, Y[:, 0], 'yellow')
        plt.plot(t, Y[:, 1], color = 'orange')
        plt.plot(t, Y[:, 2], 'red')
        plt.plot(t, Y[:, 3], 'green')
        plt.plot(t, Y[:, 4], color = 'black')
        plt.plot(t, Y[:, 5], color = "#47144873")
        plt.plot(t, Y[:, 6], color = "#DA80D169")
        plt.plot(t, Y[:, 7], color = 'blue')

total = np.sum(Y, axis=1)
plt.plot(t, total)


plt.xlabel("tid")
plt.ylabel("population")
plt.legend()
plt.show()





