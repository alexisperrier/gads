# ----------------------------------------------------------------------------
#  Lesson 9: Classification with Logistic Regression on the Default dataset
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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

