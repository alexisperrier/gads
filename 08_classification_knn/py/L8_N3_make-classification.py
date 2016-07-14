# Difficult set

# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification

# Impact of correlation on classification score


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

from sklearn import neighbors

import matplotlib.pyplot as plt
%matplotlib

np.random.seed(8)

from sklearn.datasets import make_classification


# Euclidian distance
parameters = {'n_neighbors' : np.arange(1,16,5) }
knn = neighbors.KNeighborsClassifier()
clf = GridSearchCV(knn, parameters)


# non correlated
np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_redundant=0, n_classes=2, flip_y=0.01)
pd.DataFrame(X).corr()
clf.fit(X, y)
clf.best_score_
clf.best_params_
# => 0.92, 6

# correlated
np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_redundant=3, n_classes=2, flip_y=0.01)
pd.DataFrame(X).corr()
clf.fit(X, y)
clf.best_score_
clf.best_params_
# => 0.92, 6

# n_repeated
np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_repeated=3, n_redundant=0, n_classes=2, flip_y=0.01)
pd.DataFrame(X).corr()
clf.fit(X, y)
clf.best_score_
clf.best_params_
# => 0.94, 6

# Increase the number of repeated
np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_repeated=1, n_redundant=0, n_classes=2, flip_y=0.01)
clf.fit(X, y)
clf.best_score_
clf.best_params_

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_repeated=2, n_redundant=0, n_classes=2, flip_y=0.01)
clf.fit(X, y)
clf.best_score_
clf.best_params_

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_repeated=3, n_redundant=0, n_classes=2, flip_y=0.01)
clf.fit(X, y)
clf.best_score_
clf.best_params_

# Increase the number of redundant

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_redundant=1, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("r:1 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_redundant=2, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("r:2 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_redundant=3, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("r:3 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )

# Increase the number of features keeping informative set

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("f:3/3 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=4, n_informative=3, n_redundant=0, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("f:4/3 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=0, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("f:5/3 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )

# Add correlated, keep f = i + r => no noisy feature

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=4, n_informative=3, n_redundant=1, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("f/r:4/3 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=2, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("f/r:5/2 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )

np.random.seed(8)
X, y = make_classification(n_samples=500, n_features=6, n_informative=3, n_redundant=3, n_classes=2, flip_y=0.01)
clf.fit(X, y)
print("f/r:6/3 best_score: %0.4f best_param %s   "% (clf.best_score_, clf.best_params_) )



