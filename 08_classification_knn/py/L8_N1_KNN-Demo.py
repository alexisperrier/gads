# Notebook L8 N1 Iris dataset Demo: KNN In Action

from sklearn.metrics import accuracy_score
import sklearn.datasets

iris = load_iris()
X = iris.data
y = iris.target

# shuffle dataset
np.random.seed(8)
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# split Train Test
X_train = X[:100]; y_train = y[:100]; X_test = X[100:]; y_test = y[100:]

# K = 5
clf = neighbors.KNeighborsClassifier(5)
clf.fit(X_train, y_train)

y_hat= clf.predict(X_test)

#
acc = []
for k in range(2,20, 1):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    score = accuracy_score(y_test, y_hat )
    acc.append(score)
    print("k: %s accuracy: %0.2f missed %s"% (k, score, np.count_nonzero(y_hat - y_test))   )

