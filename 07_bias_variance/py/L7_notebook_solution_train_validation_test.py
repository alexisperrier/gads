
MSE = []
alphas = [0.001, 0.005, 0.01,0.025, 0.05, 0.075, 0.1, 0.5, 1, 2]
best_score = 100
best_lm = linear_model.Ridge()

for a in alphas:
    lm = linear_model.Ridge(alpha = a)
    lm.fit(X_train, y_train)
    # predict on validation set
    y_hat = lm.predict(X_valid)
    score = mean_squared_error(y_valid, y_hat)
    if score < best_score:
        best_lm = lm
        best_score = score
        print("score : %s best score %s"% (score, best_score))
        print(best_lm)
    MSE.append( mean_squared_error(y_valid, y_hat) )

plt.plot(np.log(alphas), MSE)
plt.xlabel('log(alpha)')
plt.ylabel('MSE')

print('----\n Best Score: %s'% best_score)
print('Model:')
print(best_lm)
print(MSE)