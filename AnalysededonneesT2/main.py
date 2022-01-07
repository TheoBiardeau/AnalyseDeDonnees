import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.transpose(pd.read_csv("notes.csv", header=0, index_col=0, sep=";"))
nomi = list(X.index)
nomv = list(X.columns)
Data = X.to_numpy()
cor = np.zeros((5,5))
def histo():
    for i in range(0, len(nomv)):
        plt.hist(Data[:, i], 11)
        plt.title(nomv[i])
        plt.show()


def point():
    for i in range(0, len(Data)):
        plt.scatter(Data[i, 0], Data[i, 1])
        plt.text(Data[i, 0], Data[i, 1], nomi[i])
    plt.show()


def displayacp():
    acp = PCA()
    cc = acp.fit_transform(X)
    inertia = acp.explained_variance_
    ac = acp.components_
    vp = np.var(Data, axis=0)
    for i in range(0, len(inertia)):
        plt.scatter(i+1,inertia[i])
    plt.title("Intertie de l'ACP")
    plt.show()
    fig, ax = plt.subplots()
    ax.scatter(cc[:,0],cc[:,1])
    ax.axhline(y=0, color='k',)
    ax.axvline(x=0, color='k')
    ax.grid(True, which='both')
    plt.title("E1uE2")
    plt.show()
    fig, ax = plt.subplots()
    ax.scatter(cc[:, 0], cc[:, 2])
    ax.axhline(y=0, color='k', )
    ax.axvline(x=0, color='k')
    ax.grid(True, which='both')
    plt.title("E2uE3")
    plt.show()
    for i in range (0, 5):
        for j in range(0, 5):
            cor[j,i] = (ac[i,j]) * (inertia[j]/vp[i])**0.5
    print(cor)
displayacp()
