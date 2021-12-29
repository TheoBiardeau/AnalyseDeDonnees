import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#Ouverture de l'image et reshape
img = np.float32(mpimg.imread('visage.bmp'))
img = np.reshape(img,(256**2,3))

#Codage
kmeans = KMeans(n_clusters=16, n_init=3, init='k-means++')
kmeans.fit(img)
cluster_labels = kmeans.labels_
imgn = kmeans.cluster_centers_[kmeans.labels_[:],:]

#reshape et affichage de l'image
imgn = np.reshape(imgn,(256,256,3))
imgn = imgn / 255
plt.imshow(imgn)
plt.show()