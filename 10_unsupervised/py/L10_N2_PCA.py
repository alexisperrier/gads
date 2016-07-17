# ----------------------------------------------------------------------------
#  Lesson 10.2: Unsupervised - PCA
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
%matplotlib
import seaborn as sns
plt.style.use('fivethirtyeight')
np.random.seed(8)

# ----------------------------------------------------------------------------
#  A look at the Caravan dataset
# ----------------------------------------------------------------------------

df = pd.read_csv('../../datasets/Caravan.csv', index_col = False)
df = df.drop(['Unnamed: 0','Purchase'], axis=1)

X = df.values

# scale
scale = StandardScaler()
X = scale.fit_transform(X)

# Covariance matrix
covmat = X.T.dot(X)
covmat.shape

# Eigenvalues
from scipy import linalg
evs, evmat = linalg.eig(covmat)
evs = evs.astype(float)

evs = evs/ (len(evs)**2)

fig, ax = plt.subplots(figsize = (12,12)  )

plt.plot(evs/max(evs), label="Relative Importance")
plt.title("Importance of eigenvalues ordered")
plt.xlabel("Eigenvalues ordered")
plt.plot(np.cumsum(evs)/ max(np.cumsum(evs)), label="Cumulative Importance")
plt.grid()
plt.legend(loc='Best')

plt.annotate("66 %", (22, 0.66))
plt.annotate("X", (20, 0.66))
plt.annotate("82 %", (32, 0.82))
plt.annotate("X", (30, 0.82))


