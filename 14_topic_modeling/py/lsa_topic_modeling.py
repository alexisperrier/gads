import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from time import time
import matplotlib.pyplot as plt
import operator

# Load data
n_clusters = 5
categories   = ['sci.space','comp.graphics', 'sci.med', 'rec.motorcycles', 'rec.sport.baseball']
dataset = fetch_20newsgroups(subset='train',  categories=categories, shuffle=True, random_state=42)

print("categories:%s"% dataset.target_names)

# TfIdf
vectorizer  = TfidfVectorizer(stop_words = 'english',
                            max_features=500, use_idf=True,
                            strip_accents='ascii')
X = vectorizer.fit_transform(dataset.data)
X.shape

svd = TruncatedSVD(n_components=5,  algorithm='randomized',  n_iter=10, random_state=42)
svdX = svd.fit_transform(X)

nlzr = Normalizer(copy=False)
svdX = nlzr.fit_transform(svdX)

# Clustering
km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=4, verbose=False, random_state= 10)
km.fit(svdX)

print(" --------------------- ")
print("   Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(svdX, km.labels_, sample_size=1000))
print(" --------------------- ")

# Array mapping from words integer indices to actual words
terms   = vectorizer.get_feature_names()

n_out = 20
n_weight = 5

for k in range(n_components):
    idx = {i:abs(j) for i, j in enumerate(svd.components_[k])}
    sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
    weight = np.mean([ item[1] for item in sorted_idx[0:n_weight] ])
    print("T%s)" % k, end =' ')
    for item in sorted_idx[0:n_out-1]:
        print( " %0.3f*%s"  % (item[1] , terms[item[0]]) , end=' ')
    print()


# to plot the documents and clusters centers
# Only relevant for K = 2

y_pred  = km.predict(svdX)
centers = km.cluster_centers_

plot_clusters(svdX, y_pred, centers)


