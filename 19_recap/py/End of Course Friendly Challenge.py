import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(88)

# load train and test datasets
df  = pd.read_csv('../data/boston_train.csv', index_col = 'ID')
df_test  = pd.read_csv('../data/boston_test.csv', index_col = 'ID')

# drop rows with missing values
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# X_train, y_train, X_test, y_test
X_train = df_train.drop(['MEDV'], axis=1).values
y_train = df_train.MEDV

X_test = df_test.drop(['MEDV'], axis=1).values
y_test = df_test.MEDV

# Simple basic Random Forest regressor
clf = RandomForestRegressor(random_state=0, n_estimators=100)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

print("Baseline score (Random Forest) to beat is: %0.2f "% mean_squared_error(y_hat, y_test) )
