import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.transpose(pd.read_csv("notes.csv", header=0, index_col=0, sep=";"))

nomi = list(X.index)
nomv = list(X.columns)
Data = X.to_numpy().T

"""Data = np.array([[89,114,111,44.,10.4,57.,75.,106.,10.,],
                 [50.,58.5,75.4,5.3,0.,20.9,23.9,55.1,0.9]], dtype=float)"""

def my_acp (X):
    #Algorytme to realyze acp. Warning, study varaible are on x axis (n)


    #setup all variable
    n , p = np.shape(X)
    average_value = np.zeros(n)
    C = np.zeros((n,p))
    inertia = np.zeros(n)
    tc = np.zeros((n, p))
    tc[:,:] = X[:,:]

    #compute average value of each ligne
    for i in range (n):
        average_value[i] = np.average(X[i,:])

    #compute matrix centré
    for i in range(n):
        tc[i,:] = tc[i,:] - average_value[i]
    #compute variance covariance matrix
    Mcov = np.cov(tc)
    #compute eigen value and eigen vector
    lambT,uT = np.linalg.eigh(Mcov)
    lamb = lambT[::-1]
    u = np.flip(uT, axis=1)
    #compute intertia
    for i in range (n):
        inertia[i] = lamb[i]/(np.sum(lamb))
    #compute projection on factoriel axe
    C = np.dot(tc.T, u)
    C =C*-1
    return (lamb,C,u)

lamb, C, u =my_acp(Data)

plt.figure()
plt.scatter(C[:,0],C[:,1])
plt.show()