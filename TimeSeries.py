# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:55:12 2018

@author: JDT
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 19:32:02 2018

@author: JDT
"""

# Importing the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import Series


# Importing the dataset
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
dataset = pd.read_csv('prices.csv', parse_dates=['Ticker'], index_col='Ticker',date_parser=dateparse)
print(dataset.dtypes)
dataset.index
dataset.isnull().values.any()

# Plotting the Returns Data
plt.plot(dataset)
plt.ylabel('Prices')
plt.xlabel('Dates')
plt.title('S&P 500 Index Price Data' )
plt.show()





# Perform Dickey - Fuller Test
from statsmodels.tsa.stattools import adfuller
def ts(timeseries):
    result = adfuller(timeseries, autolag = 'AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
       
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')

# ADF Test
X = Series.from_csv('prices.csv', header=0)
ts(X)


# Eliminating Trend and Seasonality
X_log = np.log(dataset['SPY'])
plt.plot(X_log)
X_log_diff = X_log - X_log.shift(periods=1)



# Dropping missing values
X_log_diff.dropna(inplace=True)

# ADF test
ts(X_log_diff)


# Constructing PACF and ACF
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(X_log_diff)
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(X_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(X_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


lag_pacf = pacf(X_log_diff, method='ols')
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(X_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(X_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# Decomposing to check for seasonality of Time Series
import plotly
plotly.tools.set_credentials_file(username='jayverdetorres', api_key='Yk0NP6h9TDLey8gvl4h8')
from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(dataset, model='multiplicative')
fig = result.plot()
plot_mpl(fig)



import warnings
import itertools
import statsmodels.api as sm


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 6)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

bestAIC = np.inf
bestParam = None
bestSParam = None


warnings.filterwarnings("ignore") # specify to ignore warning messages

# Grid Search for finding optimal ARIMA model
for param in pdq:
        try:
            mod = sm.tsa.ARIMA(dataset,order=param)

            results = mod.fit()
            
            #if current run of AIC is better than the best one so far, overwrite it
            if results.aic<bestAIC:
                bestAIC = results.aic
                bestParam = param
            print('ARIMA{}- AIC:{}'.format(param, results.aic))
           
        except:
            continue
        
        print('the best ones are:',bestAIC,bestParam)



from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(dataset, order=(3, 1, 3))  
results_ARIMA = model.fit() 

print(results_ARIMA.summary()) 
plt.plot(X_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()

# Forecasting tomorrow's price
forecast = results_ARIMA.forecast(steps = 1)[0]





# Plotting Fitted Values vs Returns Data
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(X_log.ix[0], index=X_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(dataset , label = 'Returns Data')
plt.plot(predictions_ARIMA, label =  'Fitted Values' )
plt.ylabel('Prices')
plt.xlabel('Dates')
plt.title('Returns Data vs Fitted Values on 10^9 scale' )
plt.legend()
plt.show()







