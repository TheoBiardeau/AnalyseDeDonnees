import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = np.float32(mpimg.imread('visage.bmp'))
img = np.reshape(img,(256**2,3))
kmeans = KMeans(n_clusters=16, n_init=3, init='k-means++')
kmeans.fit(img)
cluster_labels = kmeans.labels_

imgn = kmeans.cluster_centers_[kmeans.labels_[:],:]

imgn = np.reshape(imgn,(256,256,3))
imgn = imgn / 255
print(np.shape(imgn))
print(imgn)
plt.imshow(imgn)
plt.show()