import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

from sklearn import neighbors

import matplotlib.pyplot as plt
%matplotlib


from sklearn.datasets import make_classification
from sklearn import cross_validation

clf = neighbors.KNeighborsClassifier(3)

sco = []
for n in np.arange(4,25):
    np.random.seed(8)
    X, y = make_classification(n_samples=100, n_features=n, n_informative=n, n_redundant=0, n_repeated = 0, n_classes=5, flip_y=0.01)
    scores = cross_validation.cross_val_score( clf, X, y, cv=16)
    sco.append(scores.mean())
    print("n: %s score: %0.4f  (+/- %0.2f) "% (n, scores.mean() , scores.std() * 2 ) )


fig = plt.figure(figsize=(9,9))

plt.plot(np.arange(4,25), sco)

lm = linear_model.LinearRegression()
lm.fit(np.arange(4,25).reshape(-1,1), sco)
y_hat = lm.predict(np.arange(4,25).reshape(-1,1))
plt.plot(np.arange(4,25), y_hat)
plt.ylabel('Score')
plt.xlabel('Number of dimension')
plt.title('The Curse of dimensions')



clf = neighbors.KNeighborsClassifier(5)

sco = []
dimensions = np.arange(5,260, 5)
for dim in dimensions:
    np.random.seed(8)
    X, y = make_classification(n_samples=1000, n_features=dim,
        n_redundant = 0,
        n_informative = dim ,
        n_clusters_per_class =1,
        n_classes=2,
        flip_y=0.01
        )
    scores = cross_validation.cross_val_score( clf, X, y, cv=16)
    sco.append(scores.mean())
    print("n: %s score: %0.4f  (+/- %0.2f) "% (dim, scores.mean() , scores.std() * 2 ) )


fig = plt.figure(figsize=(9,9))

plt.plot(dimensions, sco)

lm = linear_model.LinearRegression()
lm.fit(dimensions.reshape(-1,1), sco)
y_hat = lm.predict(dimensions.reshape(-1,1))
plt.plot(dimensions, y_hat)
plt.ylabel('Score')
plt.xlabel('Number of dimension')
plt.title('The Curse of dimensions')



