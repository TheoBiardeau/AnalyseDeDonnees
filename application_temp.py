import pandas as pd
from scipy.cluster.hierarchy import linkage as CAH
from scipy.cluster.hierarchy import fcluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram

data_temperature_raw = pd.read_csv("temperatures.csv", sep=";", decimal=".", header=0, index_col=0)
n = len(data_temperature_raw)
data_temperature = data_temperature_raw.drop(columns=['Region', 'Moyenne', 'Amplitude', 'Latitude', 'Longitude'])


def CAHd(data_temp):
    Z_complete = CAH(data_temp, method='complete', metric='euclidean')
    dendrogram(Z_complete, labels=data_temp.index, color_threshold=Z_complete[-2, 2])
    groupes_cah = fcluster(Z_complete, t=Z_complete[-3, 2], criterion='distance')
    plt.title("complete_methode")
    plt.show()
    print(groupes_cah)

    Coord = data_temperature_raw.loc[:, ['Latitude', 'Longitude']].values
    # Cette ligne permet d’extraire les coordonnees
    plt.scatter(Coord[:, 1], Coord[:, 0], c=groupes_cah, s=20, cmap='viridis')
    # On place les points
    nom_ville = list(data_temp.index)
    for i, txt in enumerate(nom_ville):
        plt.annotate(txt, (Coord[i, 1], Coord[i, 0]))  # On place le nom des villes
    plt.show()


def CAH_Inert(data_temp):

    mI = []
    nb_class = []
    for i in range(1, 8):
        kmeans = KMeans(n_clusters=i, n_init=3, init='k-means++')
        kmeans.fit(data_temp)
        mI.append(kmeans.inertia_)
        nb_class.append(i)
    plt.plot(nb_class, mI)
    plt.title("Inertie")
    plt.show()


def CAH_Coord (data_temp) :
    kmeans = KMeans(n_clusters=3, n_init=3, init='k-means++')
    kmeans.fit(data_temp)
    cluster_labels = kmeans.labels_
    print(cluster_labels)
    Coord = data_temperature_raw.loc[:, ['Latitude', 'Longitude']].values
    # Cette ligne permet d’extraire les coordonnees
    plt.scatter(Coord[:, 1], Coord[:, 0], c=cluster_labels, s=20, cmap='viridis')
    # On place les points
    nom_ville = list(data_temp.index)
    for i, txt in enumerate(nom_ville):
        plt.annotate(txt, (Coord[i, 1], Coord[i, 0]))  # On place le nom des villes
    plt.show()


CAHd(data_temperature)
CAH_Inert(data_temperature)
CAH_Coord(data_temperature)