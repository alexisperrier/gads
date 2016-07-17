# ----------------------------------------------------------------------------
#  Lesson 11.2: Support Vector Machine
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC


import matplotlib.pyplot as plt
%matplotlib
np.random.seed(8)


# Generating test data
np.random.seed(8)
X = np.random.randn(200,2)
X[:100] = X[:100] +2
X[101:150] = X[101:150] -2
y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)

plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2');

# grid search SVM with different kernels
parameters = [{'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.5, 1,2,3,4]}]
clf = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10, scoring='accuracy')


