import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score


def generate():
    # On fixe une seed pour toujour avoir la même génération de point
    np.random.seed(1500)

    # On définit nos paramètres
    mean_1 = np.array([2, 2])
    cov_1 = np.array([[2, 0], [0, 2]])
    mean_2 = np.array([-4, -4])
    cov_2 = np.array([[6, 0], [0, 6]])

    # Génération des nuages de point avec numpy
    N1 = (np.random.multivariate_normal(mean_1, cov_1, 128))
    N2 = (np.random.multivariate_normal(mean_2, cov_2, 128))

    # affichage des deux nuages de points
    plt.figure()
    plt.plot(N1[:, 0], N1[:, 1], "o", c="black", label='Individu appartenant à N1')
    plt.plot(N2[:, 0], N2[:, 1], "o", c="red", label='Individu appartenant à N2')
    plt.title("N1 et N2")
    plt.legend()
    plt.show()
    return (N1, N2)


def coalescence(x, K, g):
    _Llen , _Clen = np.shape(x)
    clas = np.zeros(_Llen)
    distance = np.zeros(([_Llen,K]))
    g2 = np.zeros([K,2])
    tempx = 0
    tempy = 0
    nb = 0

    while 1 :
        print("it")
        for i in range(_Llen):
            for j in range (K):
                distance[i,j] = np.sqrt(abs(x[i,0] - g[j,0])+abs(x[i,1] - g[j,1]))

        for i in range(_Llen):
            clas[i] = np.argmin([distance[i,:]])

        for i in range (K):
            for j in range (_Llen):
                if clas[j] == i :
                    tempx = tempx + x[j,0]
                    tempy = tempy + x[j,1]
                    nb = nb + 1
            g2[i,0] = tempx / nb
            g2[i,1] = tempy / nb
            tempx = tempy = nb = 0

        print(g2 == g)
        if (np.array_equal(g,g2)) :
            break
        else:
            g[:,:] = g2[:,:]

    return (clas,g2)




N1, N2 = generate()
data = np.concatenate((N1, N2), axis=0)
g = np.array([[-2.,2.],
              [-6.,-2.],
              [0,0]])
cluster_labels ,g2  = coalescence(data,3,g)
print("classe" , cluster_labels , "centroid", g2)

for i in range(0, 3):
    plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], label='Individu du cluster n°' + str(i + 1))
plt.legend()
plt.show()

CX = np.zeros([128])
CY = np.ones([128])
C = np.concatenate((CX, CY))
print("score :",adjusted_rand_score(cluster_labels, C)*100)