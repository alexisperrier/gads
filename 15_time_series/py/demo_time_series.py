import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

df = pd.read_csv('../data/Dow-Jones.csv', parse_dates = ['Date'], infer_datetime_format = True)


df = pd.read_csv('../data/Dow-Jones.csv', parse_dates=['Date'], index_col='Date', infer_datetime_format = True)

# Rolling mean
df['Rolling'] =  df.Value.rolling(window=7).mean() + 1000

fig= plt.figure(figsize=(12,9))
plt.plot(df.Date, df.Value, label = 'Value')
plt.plot(df.Date, df.Rolling, label = 'Smoothed')
plt.legend(loc='best')

# Ewma

fig= plt.figure(figsize=(12,9))
for alpha in [0.01, 0.1, 0.5, 1]:
    plt.plot(df.Date,  df.Value.ewm(alpha=alpha,ignore_na=False,adjust=True,min_periods=0).mean()  + alpha * 1000  , label = alpha)
plt.legend(loc='best')

