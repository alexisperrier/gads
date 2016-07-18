import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

%matplotlib
import sklearn.datasets
from sklearn.datasets import load_iris
from sklearn import tree

df = pd.read_csv('../../datasets/Caravan.csv', index_col = False)

# 348 Yes samples for 5822 No
# Build a test / train that respects this imbalance
# 80/20 => test: 70 Yes 1095 No train 278 Yes and 4379

# shuffle
np.random.seed(88)
df = df.sample(frac=1).reset_index(drop=True)


df_test     = df[df.Purchase == 'Yes'][:70]
df_train    = df[df.Purchase == 'Yes'][70:]

df_test = df_test.append(df[df.Purchase == 'No'][:1095])
df_train = df_train.append(df[df.Purchase == 'No'][1095:])

# verification
df_test.Purchase.value_counts()
df_train.Purchase.value_counts()

df_test = df_test.sample(frac=1).reset_index(drop=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)

y_train = df_train.Purchase.factorize()[0]
X_train = df_train.drop(['Purchase'], axis=1)

y_test = df_test.Purchase.factorize()[0]
X_test = df_test.drop(['Purchase'], axis=1)


rf = RandomForestClassifier(criterion='gini',
    bootstrap=False,
    random_state=88, )

parameters = {
    'n_estimators': [200],
    'max_depth': [10, None],
    'min_samples_split' :[2, 20],
    'max_features' : ['auto'],
}

# clf = GridSearchCV(rf, parameters, cv=5, scoring='roc_auc')
clf = GridSearchCV(rf, parameters, cv=5, scoring='f1')

clf.fit(X_train,y_train)
# confusion matrix of best model
clf.grid_scores_

y_hat =  clf.predict(X_test)

# fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test))

# print( accuracy_score(y_hat,y_test))
# print( roc_auc_score(y_hat,y_test))
# 94% accuracy!!!

# print(classification_report(y_test, y_hat))
print( confusion_matrix(y_test, y_hat))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,0])

plt.plot(fpr, tpr)

