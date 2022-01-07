import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

X = np.transpose(pd.read_csv("notes.csv", header=0, index_col=0, sep=";"))

nomi = list(X.index)
nomv = list(X.columns)
Data = X.to_numpy().T

"""Data = np.array([[89,114,111,44.,10.4,57.,75.,106.,10.,],
                 [50.,58.5,75.4,5.3,0.,20.9,23.9,55.1,0.9]], dtype=float)"""


def my_acp(X):
    # Algorithme poiur réaliser l'ACP
    # Les données étudiées sont sur l'axe x

    # Initialisation de toutes les variables
    n, p = np.shape(X)
    average_value = np.zeros(n)
    C = np.zeros((n, p))
    inertia = np.zeros(n)
    tc = np.zeros((n, p))
    tc[:, :] = X[:, :]

    # Calcul de la moyenne pour chaque ligne
    for i in range(n):
        average_value[i] = np.average(X[i, :])

    # Calcul de la matrice centrée
    for i in range(n):
        tc[i, :] = tc[i, :] - average_value[i]

    # Calcul del la matrice de variance covariance
    Mcov = np.cov(tc)

    # Calcul des valeurs et vecteurs propres
    lambT, uT = np.linalg.eigh(Mcov)
    lamb = lambT[::-1]
    u = np.flip(uT, axis=1)

    # Calcul de l'inertie
    for i in range(n):
        inertia[i] = lamb[i] / (np.sum(lamb))

    # Projections sur les axes factoriels
    C = np.dot(tc.T, u)
    C = C * -1
    for i in range(2,n):
        C[:, i] = C[:, i] * -1
    return (lamb, C, u)


lamb, C, u = my_acp(Data)

acp = PCA()
cc = acp.fit_transform(X)
fig, ax = plt.subplots()
ax.scatter(cc[:, 1], cc[:, 2], c="r", marker="2", label = "ACP sklearn")
plt.scatter(C[:, 1], C[:, 2], c="b", marker="1", label = "My ACP")
ax.axhline(y=0, color='k', )
ax.axvline(x=0, color='k')
ax.grid(True, which='both')
plt.title("E1uE2")
plt.legend()
plt.show()
#On peut observer que les points calculer avec l'ACP et notre ACP sont les mêmes, le programme fonctionne
