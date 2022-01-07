import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import random


# Fonction permetttant de générer aléatoirement un nuage de points
def generate():
    # On fixe une seed pour toujour avoir la même génération de point
    np.random.seed(1500)

    # On définit nos paramètres
    mean_1 = np.array([2, 2])
    cov_1 = np.array([[2, 0], [0, 2]])
    mean_2 = np.array([-4, -4])
    cov_2 = np.array([[6, 0], [0, 6]])

    # Génération des nuages de points avec numpy
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


# Fonction qui réalise l'algorithme  de coalescence
def coalescence(x, K, g):
    # On définit toutes nos variables
    _Llen, _Clen = np.shape(x)
    clas = np.zeros(_Llen)
    distance = np.zeros(([_Llen, K]))
    g2 = np.zeros([K, 2])
    tempx = 0
    tempy = 0
    nb = 0

    # Boucle qui réalise l'algorithme  jusqu'à convergence des centres de gravité
    while 1:

        # On calcule nos distances entre les centres de gravité
        for i in range(_Llen):
            for j in range(K):
                distance[i, j] = np.sqrt(abs(x[i, 0] - g[j, 0]) + abs(x[i, 1] - g[j, 1]))

        # On attribut à nos classes le numéro de leur cluster
        for i in range(_Llen):
            clas[i] = np.argmin([distance[i, :]])

        # On calcule les nouveaux centres de gravité
        for i in range(K):
            for j in range(_Llen):
                if clas[j] == i:
                    tempx = tempx + x[j, 0]
                    tempy = tempy + x[j, 1]
                    nb = nb + 1
            g2[i, 0] = tempx / nb
            g2[i, 1] = tempy / nb
            tempx = tempy = nb = 0
        if (np.array_equal(g, g2)):
            break
        else:
            # Condition d'arrêt : Quand les centres de gravité N-1 et N sont égaux
            g[:, :] = g2[:, :]

    return (clas, g2)


# Fonction qui réalise le choix aléatoire des centres de gravité
def randomChoice(x, K):
    n, p = np.shape(x)
    g = np.zeros((K, 2))
    for i in range(K):
        rd = random.randint(0, n)
        g[i, :] = x[rd, :]
    return (g)


# Code de test et de vérification
N1, N2 = generate()
data = np.concatenate((N1, N2), axis=0)
g = randomChoice(data, 2)
cluster_labels, g2 = coalescence(data, 2, g)
print(g2)
# Affichage des points avec leur cluster
for i in range(0, 2):
    plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], label='Individu du cluster n°' + str(i + 1))
plt.legend()
plt.show()

# Score de précision
CX = np.zeros([128])
CY = np.ones([128])
C = np.concatenate((CX, CY))
print("score :", adjusted_rand_score(cluster_labels, C) * 100)
