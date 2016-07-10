import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

from sklearn import neighbors
from sklearn import datasets
import matplotlib.pyplot as plt
%matplotlib


from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target


# XX = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
XX = pca.transform(X)
XX
print(pca.explained_variance_ratio_)

# Visualize Iris:

plt.scatter(XX[:,0], X[:,1], c = y, cmap = plt.cm.coolwarm, label = y)
