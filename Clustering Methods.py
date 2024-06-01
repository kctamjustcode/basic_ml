import sklearn
import math
import numpy as np
import scipy

from sklearn.cluster import KMeans, DBSCAN, Birch
from scipy.spatial import distance

a = (1, 2, 3)
b = (4, 5, 6)
dst = distance.euclidean(a, b)
print(dst)

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

kmeans = KMeans(n_clusters=2).fit(X)
#kmeans.predict([[0, 0], [12, 3]])
print(kmeans.labels_)

DB = DBSCAN(eps=3, min_samples=2).fit(X)
print(DB.labels_)
brc = Birch(n_clusters=2).fit(X)
print(brc.labels_)

from sklearn.metrics.cluster import normalized_mutual_info_score
print(normalized_mutual_info_score(brc.labels_,kmeans.labels_))
