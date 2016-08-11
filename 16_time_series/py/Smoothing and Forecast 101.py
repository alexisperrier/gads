import matplotlib.pyplot as plt
import pandas as pd

%matplotlib

ts = pd.read_csv('../data/tree-rings.csv', parse_dates = ['year'], index_col = 'year', infer_datetime_format = True)
ts.head()

ts['ma'] =  ts.rings.rolling(window=7).mean()


# most simple forecasting
ts['simple'] = None
ts.simple[1:ts.shape[0]] = ts.rings[0:ts.shape[0]-1]

# Metrics

from sklearn.metrics import mean_squared_error

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mad(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / len(y_true)

y_true = ts.rings[1:ts.shape[0]]
y_pred = ts.simple[1:ts.shape[0]]

print("MAPE: %0.2f"% mape(y_true, y_pred))
print("MAD: %0.5f"% mad(y_true, y_pred))
print("MSE: %0.2f"% mse(y_true, y_pred))

# Moving Average

ts['cma'] = ts.rings.rolling(window=7, center= True).mean()

# Exercize 1

fig= plt.figure(figsize=(12,9))
plt.plot(ts.index, ts.rings, label = 'Ring size', alpha = 0.5)
for i in range(2,30, 5):
    plt.plot(ts.index, ts.rings.rolling(window=i, center=True).mean() + i , label = 'MA: %d'% i, linewidth=2)
plt.xlim(xmin = min(ts.index)-1, xmax = max(ts.index) +1 )
plt.legend(loc='best')

# Window sizes

results = []
for i in range(2,26, 3):
    ts['cma'] = ts.rings.rolling(window=i, center= True).mean()
    cond = ~ts.cma.isnull()
    y_true = ts[cond].rings
    y_pred = ts[cond].cma
    results.append([ i, mse(y_true, y_pred), mad(y_true, y_pred), mape(y_true, y_pred) ]  )

res_ma = pd.DataFrame(results, columns = ['window_size','MSE', 'MAD', 'MAPE'], )

fig= plt.figure(figsize=(18,6))
plt.subplots(311)
plt.plot(res_ma.window_size, res_ma.MSE)
plt.subplots(312)
plt.plot(res_ma.window_size, res_ma.MAD)
plt.subplots(313)
plt.plot(res_ma.window_size, res_ma.MAPE)
plt.show()

# EWMA

fig= plt.figure(figsize=(12,9))
plt.plot(ts.index, ts.rings, label = 'Ring size', alpha = 0.5)
for i in  np.linspace(1, 0.0001,10):
    plt.plot(ts.index, ts.rings.ewm(alpha = i).mean() + i* 10 , label = 'EWMA: %s'% i)
plt.xlim(xmin = min(ts.index)-1, xmax = max(ts.index) +1 )
plt.legend(loc='best')

# Autocorrelation
from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(ts.rings)

# Load dow jones
ts = pd.read_csv('../data/Dow-Jones.csv', parse_dates=['Date'], index_col='Date', infer_datetime_format = True)
ts = ts[:'2010-01-01']

autocorrelation_plot(ts.Value)

# Avg temp
ts = pd.read_csv('../data/mean-daily-temperature.csv', parse_dates=['date'], index_col='date', infer_datetime_format = True)

autocorrelation_plot(ts.temp)

# PACF
import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts.rings, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts.rings, lags=40, ax=ax2)

# -------------------------
# Stationary tests

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(ts)



