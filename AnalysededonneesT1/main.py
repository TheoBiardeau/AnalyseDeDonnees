import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage as CAH
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster


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

    # Affichage des deux nuages de points
    plt.figure()
    plt.plot(N1[:, 0], N1[:, 1], "o", c="black", label='Individu appartenant à N1')
    plt.plot(N2[:, 0], N2[:, 1], "o", c="red", label='Individu appartenant à N2')
    plt.title("N1 et N2")
    plt.legend()
    plt.show()
    return (N1, N2)


def clustering(data, K, N):
    # Clustering avec la méthode de Kmeans
    kmeans = KMeans(n_clusters=K, n_init=N, init='k-means++')
    kmeans.fit(data)
    cluster_labels = kmeans.labels_

    # Affichage des données
    for i in range(0, K):
        plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1],
                    label='Individu du cluster n°' + str(i + 1))
    plt.legend()
    plt.show()

    # On vérifie le taux de réussite
    CX = np.zeros([128])
    CY = np.ones([128])
    C = np.concatenate((CX, CY))
    print("score :", adjusted_rand_score(cluster_labels, C) * 100)


def choosegoodK(data):
    # On nomme nos variables qui vont stocker les données
    nb_class = []
    mSv = []
    mI = []

    # On réalise une boucle où nous allons noter l'inertie
    # et la silhouette à chaque itération
    for i in range(1, 7):
        kmeans = KMeans(n_clusters=i, n_init=3, init='k-means++')
        kmeans.fit(data)
        cluster_labels = kmeans.labels_
        mI.append(kmeans.inertia_)
        nb_class.append(i)
        if (i > 1):
            mSv.append(silhouette_score(data, cluster_labels))

    # on affiche nos données
    plt.plot(nb_class, mI)
    plt.title("Inertie")
    plt.show()

    plt.plot(nb_class[1:], mSv)
    plt.title("Silhouette")
    plt.show()


def CAH_complete_methode(data, K):
    # Clustering avec la méthode CAH "complete"
    Z_complete = CAH(data, method='complete', metric='euclidean')
    treshold = Z_complete[-K + 1, 2]
    d = dendrogram(Z_complete, color_threshold=treshold)
    groupes_cah = fcluster(Z_complete, t=Z_complete[-K, 2], criterion='distance')

    # Ajout de la ligne horizontale de la coupe
    plt.axhline(y=Z_complete[-K, 2], c='grey', lw=1, linestyle='dashed')
    plt.title("complete_methode")
    plt.show()

    CX = np.zeros([128])
    CY = np.ones([128])
    C = np.concatenate((CX, CY))
    print("score :", adjusted_rand_score(groupes_cah, C) * 100)


def comp(data, K):
    fig, axs = plt.subplots(2, 2)
    Z_complete = CAH(data, method='complete', metric='euclidean')
    treshold = Z_complete[-K + 1, 2]
    d = dendrogram(Z_complete, ax=axs[0, 0], color_threshold=treshold)
    groupes_cahC = fcluster(Z_complete, t=Z_complete[-K, 2], criterion='distance')
    axs[0, 0].set_title("Complete")
    plt.sca(axs[0, 0])
    plt.xticks(color='w')

    Z_complete = CAH(data, method='average', metric='euclidean')
    treshold = Z_complete[-K + 1, 2]
    e = dendrogram(Z_complete, color_threshold=treshold, ax=axs[1, 0])
    groupes_cahA = fcluster(Z_complete, t=Z_complete[-K, 2], criterion='distance')
    axs[1, 0].set_title("Average")
    plt.sca(axs[1, 0])
    plt.xticks(color='w')

    Z_complete = CAH(data, method='single', metric='euclidean')
    treshold = Z_complete[-K + 1, 2]
    f = dendrogram(Z_complete, color_threshold=treshold, ax=axs[0, 1])
    groupes_cahS = fcluster(Z_complete, t=Z_complete[-K, 2], criterion='distance')
    axs[0, 1].set_title("Single")
    plt.sca(axs[0, 1])
    plt.xticks(color='w')

    Z_complete = CAH(data, method='ward', metric='euclidean')
    treshold = Z_complete[-K + 1, 2]
    g = dendrogram(Z_complete, color_threshold=treshold, ax=axs[1, 1])
    groupes_cahW = fcluster(Z_complete, t=Z_complete[-K, 2], criterion='distance')
    axs[1, 1].set_title("Ward")
    plt.sca(axs[1, 1])
    plt.xticks(color='w')
    plt.show()

    CX = np.zeros([128])
    CY = np.ones([128])
    C = np.concatenate((CX, CY))
    print("score Ward :", adjusted_rand_score(groupes_cahW, C) * 100)
    print("score Average :", adjusted_rand_score(groupes_cahA, C) * 100)
    print("score Single :", adjusted_rand_score(groupes_cahS, C) * 100)
    print("score Complete :", adjusted_rand_score(groupes_cahC, C) * 100)


N1, N2 = generate()
donnee = np.concatenate((N1, N2), axis=0)
clustering(donnee, 2, 3)
