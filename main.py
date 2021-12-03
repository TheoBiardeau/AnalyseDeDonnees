import numpy as np
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
nb_class = 3

mSc = []
mSv = []
mI = []
nb_class = []
for i in range(1, 8):

    X1 = (np.random.multivariate_normal(mean_1, cov_1, 128))
    X2 = (np.random.multivariate_normal(mean_2, cov_2, 128))
    data = np.concatenate((X1, X2), axis=0)
    kmeans = KMeans(n_clusters=i, n_init=3, init='k-means++')
    kmeans.fit(data)
    cluster_labels = kmeans.labels_

    """
    plt.plot(data[kmeans.labels_ == 0,0],data[kmeans.labels_ == 0,1],"o",color = "black",label = 'Individu 1')
    plt.plot(data[kmeans.labels_ == 1,0],data[kmeans.labels_ == 1,1],"*",color = "red",label = 'Individu 2')
    plt.legend()
    plt.show()
    
    
    CX = np.zeros([128])
    CY = np.ones ([128])
    C = np.concatenate((CX,CY))
    print(adjusted_rand_score(kmeans.labels_, C))
    """
    mI.append(kmeans.inertia_)
    nb_class.append(i)
    if (i > 1):
        mSv.append(silhouette_score(data, cluster_labels))


plt.plot(nb_class, mI)
plt.title("Inertie")
plt.show()

plt.plot(nb_class[1:], mSv)
plt.show()
