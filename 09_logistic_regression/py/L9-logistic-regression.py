# ----------------------------------------------------------------------------
#  Lesson 9: Classification with Logistic Regression on the Default dataset
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics  import confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib
import seaborn as sns
plt.style.use('fivethirtyeight')


np.random.seed(8)

# ----------------------------------------------------------------------------
#  Exploratory analysis of the Default Dataset
# ----------------------------------------------------------------------------

df = pd.read_csv('../data/Default.csv', index_col = False)

# 10000 rows with
print(df.shape())
print(df.head())

# Factorize / Encode the student regressor and the default target
df['default_fact']   = df.default.factorize()[0]
df['student_fact']   = df.student.factorize()[0]

# describe
print(df.describe())

# box plots balance and income vs default
fig = plt.figure(figsize=(12,9))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

c_palette = {'No':'lightblue', 'Yes':'orange'}
sns.boxplot('default', 'balance', data=df, orient='v', ax=ax1, palette=c_palette)
sns.boxplot('default', 'income', data=df, orient='v', ax=ax2, palette=c_palette)

# ----------------------------------------------------------------------------
#  Regression default ~ balance
# ----------------------------------------------------------------------------

# Add the constant term to X  [[1,x_1], [1,x_2], ..., [1,x_n], ]
# expected by statsmodel (not by scikit learn)
X_sm = sm.add_constant(df.balance)
clf = sm.Logit(df.default_fact, X_sm).fit()
print(clf.summary().tables[1])

# ==============================================================================
#                  coef    std err          z      P>|z|      [95.0% Conf. Int.]
# ------------------------------------------------------------------------------
# const        -10.6513      0.361    -29.491      0.000       -11.359    -9.943
# balance        0.0055      0.000     24.952      0.000         0.005     0.006
# ==============================================================================

# ----------------------------------------------------------------------------
#  Regression default ~ student
# ----------------------------------------------------------------------------

X_sm = sm.add_constant(df.student_fact)
clf = sm.Logit(df.default_fact, X_sm).fit()
print(clf.summary().tables[1])

# --------------------------------------------------------------------------------
# const           -3.5041      0.071    -49.554      0.000        -3.643    -3.366
# student_fact     0.4049      0.115      3.520      0.000         0.179     0.630

# ----------------------------------------------------------------------------
#  MultiRegression default ~ balance + income + student
# ----------------------------------------------------------------------------
X_sm = sm.add_constant(df[['balance', 'income', 'student_fact']])
clf = smf.Logit(df.default_fact, X_sm).fit()
print(clf.summary().tables[1])


# --------------------------------------------------------------------------------
# const          -10.8690      0.492    -22.079      0.000       -11.834    -9.904
# balance          0.0057      0.000     24.737      0.000         0.005     0.006
# income        3.033e-06    8.2e-06      0.370      0.712      -1.3e-05  1.91e-05
# student_fact    -0.6468      0.236     -2.738      0.006        -1.110    -0.184

# Why is student_fact >0 when alone and <0 when taken with balance and income?


# Probability of default for a student with a credit card balance of 1500 and income of 40k

yy = clf.params['const'] + clf.params['student_fact'] + clf.params['balance'] * 1500 +  clf.params['income'] * 40
p = np.exp(yy) / (1+ np.exp(yy))
print("Default Probability for Student: %0.4f"%  p )
# Probability of default for a student with a credit card balance of 1500 and income of 40k

yy = clf.params['const'] + clf.params['balance'] * 1500 +  clf.params['income'] * 40
p = np.exp(yy) / (1+ np.exp(yy))
print("Default Probability for Non Student: %0.4f"%  p )

# --------------------------------------------------------------------------------
#  Confusion Matrix in Scikit
# --------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix
y_true = [0,0,0,0,0,1,1,1,1,1]
y_pred = [0,0,0,1,1,0,1,1,1,1]
confusion_matrix(y_true, y_pred)

# ----------------------------------------------------------------------------
#  Logistic Regression
# ----------------------------------------------------------------------------
from sklearn.cross_validation import train_test_split

df = pd.read_csv('../data/Default.csv', index_col = False)
df['default_fact']   = df.default.factorize()[0]
df['student_fact']   = df.student.factorize()[0]

X = df[['balance', 'income', 'student_fact']]
y = df.default_fact

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=88)

lr = LogisticRegression()
parameters = {'solver': ['newton-cg','lbfgs','liblinear','sag']}

clf = GridSearchCV(lr, parameters, cv = 5)

clf.fit(X_train, y_train)

best_clf = clf.best_estimator_

y_hat = best_clf.predict(X_test)
print(confusion_matrix(y_test, y_hat) )

print("Best accuracy score for Logistic Regression: %0.4f"% accuracy_score(y_test, y_hat) )

# ----------------------------------------------------------------------------
#  K-NN
# ----------------------------------------------------------------------------

knn = KNeighborsClassifier()
parameters = {'n_neighbors': [2,5,10] }

clf = GridSearchCV(knn, parameters, cv = 5)

clf.fit(X_train, y_train)

best_knn = clf.best_estimator_

y_hat = best_knn.predict(X_test)

print(confusion_matrix(y_test, y_hat) )

print("Best accuracy score for K-NN (K:%s): %0.4f"%  (str(clf.best_params_),  accuracy_score(y_test, y_hat) ) )

# ----------------------------------------------------------------------------
#  Predict, Predict_proba and the question of the threshold
# ----------------------------------------------------------------------------

pred = pd.DataFrame( {'y_hat':y_hat, 'y_proba': y_proba[:,1] }  )

fig = plt.figure(figsize=(12,9))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
c_palette = {0:'lightblue', 1:'orange'}
sns.boxplot('y_hat', 'y_proba', data=pred[y_hat == 0], orient='v', ax=ax1, palette=c_palette)
sns.boxplot('y_hat', 'y_proba', data=pred[y_hat == 1], orient='v', ax=ax2, palette=c_palette)

# Other threshold
t = 0.4

def threshold(x):
    if x < t:
        return 0
    else:
        return 1

pred['y_hat_t'] = pred.y_proba.apply(lambda x : threshold(x)  )

print(confusion_matrix(y_test, pred.y_hat_t))
print(confusion_matrix(y_test, pred.y_hat))

print(accuracy_score(y_test, pred.y_hat_t))
print(accuracy_score(y_test, pred.y_hat))

# Look at many thresholds

for t in np.linspace(0.1, 1, 10):
    pred['y_hat_t'] = pred.y_proba.apply(lambda x : threshold(x)  )
    print(" Threshold %s, accuracy, %0.4f "% (t, accuracy_score(y_test, pred.y_hat_t)) )


# ----------------------------------------------------------------------------
#  ROC Curve http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# ----------------------------------------------------------------------------

from sklearn.metrics import roc_curve, auc


fpr, tpr, thresholds = roc_curve(y_test, pred.y_proba)

roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# with best_knn

y_proba = best_knn.predict_proba(X_test)[:,1]



