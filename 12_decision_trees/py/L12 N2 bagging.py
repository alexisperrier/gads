import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

%matplotlib
import sklearn.datasets
from sklearn.datasets import load_iris
from sklearn import tree

df = pd.read_csv('../../datasets/Hitters.csv', index_col = False)

df['League']        = df.League.factorize()[0]
df['Division']      = df.Division.factorize()[0]
df['NewLeague']     = df.NewLeague.factorize()[0]

df.dropna(subset=['Salary'], inplace = True)
df.drop(['Player'], axis = 1)


X = df.drop(['Salary'], axis = 1)
y = df.Salary

# boston dataset
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=88)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

###############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)


np.random.seed(88)

# ----------------------------------------------------------------------------
#  Basic Decision tree
# ----------------------------------------------------------------------------

np.random.seed(88)
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)


scores = cross_val_score(clf, X, y, cv=50)
print("Default Decision Tree scores: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))

# ----------------------------------------------------------------------------
#  limit the tree
# ----------------------------------------------------------------------------

clf = DecisionTreeRegressor(max_depth = 3, min_samples_split = 10)
clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)


scores = cross_val_score(clf, X, y, cv=50)
print("Constrained Decision Tree scores: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))

# ----------------------------------------------------------------------------
#  Bagging - default tree
# ----------------------------------------------------------------------------
from sklearn.ensemble import BaggingClassifier

clf = DecisionTreeRegressor()
bagging = BaggingRegressor(base_estimator = clf, n_estimators = 20, max_samples=0.5, bootstrap=True, oob_score = True)
bagging.fit(X_train, y_train)
mse = mean_squared_error(y_test, bagging.predict(X_test))
print("MSE: %.4f" % mse)

scores = cross_val_score(bagging, X, y, cv=50)
print("Bagging - Default Decision Tree scores: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))

# ----------------------------------------------------------------------------
#  Bagging - Constrained tree
# ----------------------------------------------------------------------------
clf = DecisionTreeRegressor(max_depth = 3, min_samples_split = 10)
bagging = BaggingRegressor(base_estimator = clf, n_estimators = 20, max_samples=0.5, bootstrap=True, oob_score = True)
scores = cross_val_score(bagging, X, y, cv=50)
print("Bagging - Constrained Decision Tree scores: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))

bagging.fit(X_train, y_train)
mse = mean_squared_error(y_test, bagging.predict(X_test))
print("MSE: %.4f" % mse)



# ----------------------------------------------------------------------------
#  Classficiation
# ----------------------------------------------------------------------------

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Generate a binary classification dataset.
np.random.seed(88)
X, y = make_classification(n_samples=500, n_features=25, n_clusters_per_class=1, n_informative=20, random_state=88)

clf1 = DecisionTreeClassifier()
scores = cross_val_score(clf1, X, y, cv=20)
print("Simple tree: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))
# 0.88
clf2 = DecisionTreeClassifier(max_depth = 5, min_samples_split = 10)
scores = cross_val_score(clf2, X, y, cv=20)
print("Constrained tree: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))
# 0.87

bagging1 = BaggingClassifier(base_estimator = clf1, n_estimators = 100, max_samples=0.5, bootstrap=True, oob_score = True)
bagging2 = BaggingClassifier(base_estimator = clf1, n_estimators = 10, max_samples=0.5, bootstrap=True, oob_score = True)
bagging3 = BaggingClassifier(base_estimator = clf1, n_estimators = 500, max_samples=0.5, bootstrap=True, oob_score = True)

scores = cross_val_score(bagging2, X, y, cv=20)
print("bagging 10 estimator: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))

scores = cross_val_score(bagging1, X, y, cv=20)
print("bagging 100 estimator: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))

scores = cross_val_score(bagging3, X, y, cv=20)
print("bagging 500 estimator: %0.2f (+/- %0.2f) "% (np.mean(scores),np.std(scores) ))


