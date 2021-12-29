import pandas as pd
from scipy.cluster.hierarchy import linkage as CAH
from scipy.cluster.hierarchy import fcluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram

def extraction ():
    data_temperature_raw = pd.read_csv("temperatures.csv", sep=";", decimal=".", header=0, index_col=0)
    data_temperature = data_temperature_raw.drop(columns=['Region', 'Moyenne', 'Amplitude', 'Latitude', 'Longitude'])
    return(data_temperature, data_temperature_raw)

def CAHd(data_temp, data_temperature_raw,K):
    #Execution de l'algorithme de CAH
    Z_complete = CAH(data_temp, method='complete', metric='euclidean')
    dendrogram(Z_complete, labels=data_temp.index, color_threshold=Z_complete[-K+1, 2])
    groupes_cah = fcluster(Z_complete, t=Z_complete[-K, 2], criterion='distance')
    plt.title("complete_methode")
    plt.show()


    Coord = data_temperature_raw.loc[:, ['Latitude', 'Longitude']].values
    plt.scatter(Coord[:, 1], Coord[:, 0], c=groupes_cah, s=20, cmap='viridis')
    # On place les points
    nom_ville = list(data_temp.index)
    for i, txt in enumerate(nom_ville):
        plt.annotate(txt, (Coord[i, 1], Coord[i, 0]))  # On place le nom des villes
    plt.title("Complete_methode")
    plt.show()


def Kmeans_Inert(data_temp):

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


def Kmeans_Coord (data_temp, data_temperature_raw,K) :
    # Execution de l'algorithme de Kmeans
    kmeans = KMeans(n_clusters=K, n_init=3, init='k-means++')
    kmeans.fit(data_temp)
    cluster_labels = kmeans.labels_

    #Affichage des villes selon leur coordonées géopgrpahique et leur classe
    Coord = data_temperature_raw.loc[:, ['Latitude', 'Longitude']].values
    plt.scatter(Coord[:, 1], Coord[:, 0], c=cluster_labels, s=20, cmap='viridis')
    # On place les points
    nom_ville = list(data_temp.index)
    for i, txt in enumerate(nom_ville):
        plt.annotate(txt, (Coord[i, 1], Coord[i, 0]))  # On place le nom des villes
    plt.title("Kmeans methode")
    plt.show()

data , datar = extraction()
Kmeans_Inert(data)
CAHd(data, datar,3)
Kmeans_Coord(data, datar,3)