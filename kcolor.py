import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

im = cv2.imread("...")
cv2.imshow("im", im)
vectors = im.reshape(-1, 3) # Image chargée sous forme de tableau de vecteurs 3D

km = KMeans(n_clusters = 5).fit(vectors) # Applique l'algorithme KMeans

M = np.zeros_like(vectors)
for i in range(km.n_clusters):
    ind = np.where(km.labels_ == i)
    M[ind] = km.cluster_centers_[i] # Remplace chaque pixel par le centre de sa classe

kim = M.reshape(im.shape) # Revient au format image classique
cv2.imshow("kim", kim)
cv2.imwrite("...", kim)