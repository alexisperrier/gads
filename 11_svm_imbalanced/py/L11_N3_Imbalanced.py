# ----------------------------------------------------------------------------
#  Lesson 11.3: Imbalanced Datasets
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt
%matplotlib
np.random.seed(8)

df = pd.read_csv('../../datasets/Caravan.csv', index_col = False)

# Shuffles your dataframe in-place and resets the index
df = df.sample(frac=1).reset_index(drop=True)

# Split train / test 80/20

y = df.Purchase.factorize()[0]
X = df.drop(['Purchase'], axis=1).values
scale = StandardScaler()
X = scale.fit_transform(X)

# scale

# Simplest classifier
y_hat = np.zeros(len(y))

print( accuracy_score(y_hat,y))
# 94% accuracy!!!

print(classification_report(y, y_hat))
print( confusion_matrix(y_hat,y))

# -----------------------------------------------------------------------------
# Simple svc
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
# => 273 / 75 Yes

parameters = {'C': [0.001, 0.01,0.1, 0.5,1]}

clf = GridSearchCV(SVC(kernel='rbf'), parameters, cv=5, scoring='roc_auc')
clf.fit(X_train,y_train)
# confusion matrix of best model
print(clf.grid_scores_)

y_hat = clf.predict(X_test)

print( accuracy_score(y_hat,y_test))
# 94% accuracy!!!

# print(classification_report(y_hat, y_test))
print( confusion_matrix(y_hat,y_test))
# same problem

# -----------------------------------------------------------------------------
#  Under sample
#  let's rebalance  the training set
# -----------------------------------------------------------------------------

# first let's save the original X_test and y_test
X_test_original = X_test
y_test_original = y_test

# 348 yes samples
under_df = df[df.Purchase == 'Yes']
under_df = under_df.append(df[df.Purchase == 'No'].sample(348)).sample(frac=1).reset_index(drop=True)

# under_df has 696 samples with 50/50 yes no
y = under_df.Purchase.factorize()[0]
X = under_df.drop(['Purchase'], axis=1).values
scale = StandardScaler()
X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
parameters = {'C': [0.001, 0.01,0.1, 0.5,1]}

clf = GridSearchCV(SVC(kernel='rbf'), parameters, cv=5, scoring='roc_auc')
clf.fit(X_train,y_train)
# confusion matrix of best model
clf.grid_scores_

y_hat = clf.predict(X_test)

print( accuracy_score(y_hat,y_test))
# 66% accuracy!!!

print(classification_report(y_hat, y_test))
print( confusion_matrix(y_hat,y_test))

# not great score but at least it's getting some positives
yy_hat = clf.predict(X_test_original)
print( confusion_matrix(yy_hat,y_test_original))
print( accuracy_score(yy_hat,y_test_original))
print(classification_report(yy_hat, y_test_original))
print(roc_auc_score(yy_hat, y_test_original))

# and much better on the original test dataset but not great overall

# -----------------------------------------------------------------------------
#  Over sample
#  let's over sample the Yes cases
# -----------------------------------------------------------------------------

# Load and shuffle, scale and split
df = pd.read_csv('../../datasets/Caravan.csv', index_col = False)

# let's replicate the number of yes cases by 6
tmp = df[df.Purchase == 'Yes']
for _ in range(4):
    tmp = tmp.append( df[df.Purchase == 'Yes']  )

df = df.append(tmp).sample(frac=1).reset_index(drop=True)
print(df.Purchase.value_counts())
# we now have 5474 / 2088

y = df.Purchase.factorize()[0]
X = df.drop(['Purchase'], axis=1).values
scale = StandardScaler()
X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)

parameters = {'C': [0.001, 0.01,0.1, 0.5,1]}

clf = GridSearchCV(SVC(kernel='rbf'), parameters, cv=5, scoring='roc_auc')
clf.fit(X_train,y_train)
clf.grid_scores_

y_hat = clf.predict(X_test)
print( accuracy_score(y_hat,y_test))
# 83% accuracy!!!

print(classification_report(y_hat, y_test))
print( confusion_matrix(y_hat,y_test))


# on the original set
yy_hat = clf.predict(X_test_original)
print( confusion_matrix(yy_hat,y_test_original))
print( accuracy_score(yy_hat,y_test_original))
print(classification_report(yy_hat, y_test_original))
print(roc_auc_score(yy_hat, y_test_original))
# gives interesting results

# -----------------------------------------------------------------------------
#  Create new samples with SMOTE!
#  SMOTE: synthetic minority over-sampling technique
#  let's over sample the Yes cases
# -----------------------------------------------------------------------------

df = pd.read_csv('../../datasets/Caravan.csv', index_col = False)

df = df.sample(frac=1).reset_index(drop=True)
y = df.Purchase.factorize()[0]
X = df.drop(['Purchase'], axis=1).values
scale = StandardScaler()
X = scale.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)

# original data
pca = PCA(n_components=2)
x_vis = pca.fit_transform(X_train)

from imblearn.over_sampling import SMOTE
smox, smoy = smote.fit_sample(X_train, y_train)
smox_vis = pca.transform(smox)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
palette = sns.color_palette()
axes[0].scatter(x_vis[y == 0, 0], x_vis[y == 0, 1], label="Class #0", alpha=0.5,  facecolor=palette[0], linewidth=0.15)
axes[0].scatter(x_vis[y == 1, 0], x_vis[y == 1, 1], label="Class #1", alpha=0.5,  facecolor=palette[2], linewidth=0.15)
axes[0].legend()
plt.scatter(smox_vis[smoy == 0, 0], smox_vis[smoy == 0, 1], label="Class #0", alpha=0.5,  facecolor=palette[0], linewidth=0.15)
plt.scatter(smox_vis[smoy == 1, 0], smox_vis[smoy == 1, 1], label="Class #1", alpha=0.5,  facecolor=palette[2], linewidth=0.15)
plt.show()

# train
parameters = {'C': [0.001, 0.01,0.1, 0.5,1]}

clf = GridSearchCV(SVC(kernel='rbf'), parameters, cv=5, scoring='roc_auc')
clf.fit(smox,smoy)
clf.grid_scores_
y_hat = clf.predict(X_test)

yy_hat = clf.predict(X_test)
print( confusion_matrix(yy_hat,y_test))
print( accuracy_score(yy_hat,y_test))
print(classification_report(yy_hat, y_test))
print(roc_auc_score(yy_hat, y_test))

# Much better
