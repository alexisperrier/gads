import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import seaborn as sns

%matplotlib

df  = sm.datasets.sunspots.load()

dta = pd.DataFrame(df.data['SUNACTIVITY'],  index = sm.tsa.datetools.dates_from_range('1700', '2008'), columns = ['SUNACTIVITY'])

dta.plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts.spots, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts.spots, lags=40, ax=ax2)

autocorrelation_plot(ts.spots)

# ARMA (2,0)


arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
print arma_mod20.params

print(arma_mod20.aic)
print(arma_mod20.bic)
print(arma_mod20.hqic)

# Durbin Watson

# We will use the Durbin-Watson test for autocorrelation. The Durbin-Watson statistic ranges in value from 0 to 4.
# A value near 2 indicates non-autocorrelation; a value toward 0 indicates positive autocorrelation; a value toward 4 indicates negative autocorrelation.

# analysis of residuals

sm.stats.durbin_watson(arma_mod20.resid)
# => no autocorrelation

stats.normaltest(arma_mod20.resid)

# qq plot or residuals
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(arma_mod20.resid, line='q', ax=ax, fit=True)

# Predict

predict_sunspots20 = arma_mod20.predict('1990', '2012', dynamic=True)

# plot:
ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_sunspots20.plot(ax=ax, style='r--', label='Dynamic Prediction');
ax.legend();
ax.axis((-20.0, 38.0, -4.0, 200.0));

# Metric
def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()






