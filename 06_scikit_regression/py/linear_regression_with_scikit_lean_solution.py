'''
* Load the ozone dataset, drop rows with missing values

* With scikit learn
    * train a first linear regression model Ozone ~ Temp

    Trick:
            X = df[['Temp']].values
            y = df.Ozone

    * plot the estimated and the true outcomes
    * plot the regression
    * is that a valid regression?

* Then
    * Train another model with Ozone ~ Wind

What do you observe?

'''
df = pd.read_csv('../../datasets/ozone.csv')
df = df.dropna()
df.shape


clf = linear_model.LinearRegression()
X = df[['Temp']].values
y = df.Ozone
clf.fit(X,y)
y_hat = clf.predict(X)
residue = y - y_hat


fig,[ax1,ax2, ax3] = plt.subplots(1,3,figsize=(18,6))

# True outcomes in blue
ax1.plot(X, y,'.', c='b', label='truth')
# Predicted values in red
ax1.plot(X, y_hat,'+', c='r', label='prediction')
ax1.set_title("Prediction vs Truth")
ax1.legend(loc='best')

# Residuals
ax2.plot(X,residue,'.', c='b', label='Residuals')
ax2.set_title("Residuals")
ax2.legend(loc='best')

# QQ plot
(a, r) = stats.probplot(residue, dist="norm", plot=ax3)
