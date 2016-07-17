# ----------------------------------------------------------------------------
#  Lesson 10.1: Unsupervised - K Means Clustering
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
%matplotlib
import seaborn as sns
plt.style.use('fivethirtyeight')


np.random.seed(8)

# ----------------------------------------------------------------------------
#  The four datasets, colors and algorithms
# ----------------------------------------------------------------------------

n_samples = 1500
noisy_circles   = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons     = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs           = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure    = np.random.rand(n_samples, 2), None


# Initialize colors
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

# The algorithms
kmeans = cluster.KMeans(n_clusters=2)
dbscan = cluster.DBSCAN(eps=.2)
spectral = cluster.SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors")


# clst = kmeans, dbscan, spectral

data_titles = ['noisy_circles', 'noisy_moons', 'blobs', 'no_structure']

def cluster(clst, title):
    # Create  2x2 figure for each algorithm
    fig, ax = plt.subplots(2,2, figsize=(16,4)  )
    datasets = [noisy_circles, noisy_moons, blobs, no_structure]
    plot_num = 1
    for i, dataset in enumerate(datasets):

        X, y = dataset
        X = StandardScaler().fit_transform(X)
        clst.fit(X)
        y_pred = clst.labels_.astype(np.int)
        plt.subplot(1, 4, plot_num)

        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
        plt.title(data_titles[i])
        plot_num +=1

cluster(spectral,'Spectral')

cluster(kmeans, 'Kmeans')
cluster(dbscan, 'DBscan')


