import numpy as np
import random as rn
import matplotlib.pyplot as plt
import random
import os
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram

np.random.seed(1500)
mean_1 = np.array([2, 2])
cov_1 = np.array([[2, 0], [0, 2]])
mean_2 = np.array([-4, -4])
cov_2 = np.array([[6, 0], [0, 6]])
X1 = (np.random.multivariate_normal(mean_1, cov_1, 128))
X2 = (np.random.multivariate_normal(mean_2, cov_2, 128))
data = np.concatenate((X1, X2), axis=0)


def selection ():
    N1 = rn.choice(data)
    N2 = rn.choice(data)
    label = np.array()
    for i in range (0, len(data)):
        if data[i] - N1 >= data[i] - N2 :
            np.push(label,1)
        else:
            np.push(label,0)

