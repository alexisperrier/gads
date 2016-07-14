import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

from sklearn import neighbors

import matplotlib.pyplot as plt
%matplotlib

np.random.seed(8)

df = pd.read_csv('../data/fraudulent_balanced.csv')

# Compare best K for 2 models
X = df[ ['income','balance'] ].values
y = df.fraud

knn = neighbors.KNeighborsClassifier()

parameters = {'n_neighbors' : np.arange(2,20) }

clf = GridSearchCV(knn, parameters)
clf.fit(X, y)
# returns the mean over all K-Fold CV, the std and the parameter value
clf.grid_scores_

# Extracting the mean from grid_scores_
means = [x[1] for x in clf.grid_scores_]
plt.plot(np.arange(2,20), means)

# -------------------------------------------------------------
# second experiment: only use X = df.Balance
# and find the best K
# -------------------------------------------------------------
X = df[ ['balance'] ].values

# -------------------------------------------------------------
# 3rd experiment on Balance and Income: add the distance as a new parameter
# and find the best (K, p)
# -------------------------------------------------------------

parameters = {'n_neighbors' : np.arange(2,20), 'p': np.linspace(1,2,5) }

# -------------------------------------------------------------
# 4th experiment on Balance and Income: 
# Scaling and Normalizing with StandardScaler
# Compare before and after scaling
# -------------------------------------------------------------
X = df[ ['income','balance'] ].values

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)

clf = GridSearchCV(knn, parameters)
clf.fit(X, y)
clf.best_score_
# => 0.84
# Mean and std of X
scaler.mean_
scaler.scale_

# Scale and Transform
X = scaler.transform(X)

np.mean(X[:,0])
np.mean(X[:,1])
np.std(X[:,0])
np.std(X[:,1])

clf.fit(X, y)
clf.best_score_
# => 0.88