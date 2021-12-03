import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage as CAH
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster

np.random.seed(1500)
mean_1 = np.array([2, 2])
cov_1 = np.array([[2, 0], [0, 2]])
mean_2 = np.array([-4, -4])
cov_2 = np.array([[6, 0], [0, 6]])
nb_class = 3


X1 = (np.random.multivariate_normal(mean_1, cov_1, 128))
X2 = (np.random.multivariate_normal(mean_2, cov_2, 128))
data = np.concatenate((X1, X2), axis=0)

def complete_methode ():
    Z_complete = CAH(data,method='complete',metric='euclidean')
    treshold = Z_complete[-1,2]
    d = dendrogram(Z_complete,color_threshold=treshold)
    groupes_cah = fcluster(Z_complete, t=Z_complete[-2,2], criterion='distance')
    # Ajouter la ligne horizontale de la coupe
    plt.axhline(y=Z_complete[-2,2], c='grey', lw=1, linestyle='dashed')
    plt.title("complete_methode")
    plt.show()

def single_methode ():
    Z_complete = CAH(data,method='single',metric='euclidean')
    treshold = Z_complete[-1,2]
    d = dendrogram(Z_complete,color_threshold=treshold)
    groupes_cah = fcluster(Z_complete, t=Z_complete[-2,2], criterion='distance')
    # Ajouter la ligne horizontale de la coupe
    plt.axhline(y=Z_complete[-2,2], c='grey', lw=1, linestyle='dashed')
    plt.title("single_methode")
    plt.show()

def average_methode ():
    Z_complete = CAH(data,method='average',metric='euclidean')
    treshold = Z_complete[-1,2]
    d = dendrogram(Z_complete,color_threshold=treshold)
    groupes_cah = fcluster(Z_complete, t=Z_complete[-2,2], criterion='distance')
    # Ajouter la ligne horizontale de la coupe
    plt.axhline(y=Z_complete[-2,2], c='grey', lw=1, linestyle='dashed')
    plt.title("average_methode")
    plt.show()

def ward_methode ():
    Z_complete = CAH(data,method='ward',metric='euclidean')
    treshold = Z_complete[-1,2]
    d = dendrogram(Z_complete,color_threshold=treshold)
    groupes_cah = fcluster(Z_complete, t=Z_complete[-2,2], criterion='distance')
    # Ajouter la ligne horizontale de la coupe
    plt.axhline(y=Z_complete[-2,2], c='grey', lw=1, linestyle='dashed')
    plt.title("ward_methode")
    plt.show()

ward_methode()
complete_methode()
average_methode()
single_methode()
