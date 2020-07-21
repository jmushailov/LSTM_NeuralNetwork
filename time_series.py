# Time Series
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot

def create_series(df, xcol, datecol):
    # Create a dataframe with the features and the date time as the index
    features_considered = [xcol]
    features = df[features_considered]
    features.index = df[datecol]
    features.head()
    features.plot(subplots=True)
    return features


# X is the series to test
# log_x asks whether to log X prior to testing or not
def stationarity_test(X, log_x = "Y"):
    
    # If X isn't logged, we need to log it for better results
    if log_x == "Y":
        X = np.log(X[X>0])
    
    # Once we have the series as needed we can do the ADF test
    from statsmodels.tsa.stattools import adfuller
    dickey_fuller = adfuller(X)
    # If ADF statistic is < our 1% critical value (sig level) we can conclude it's not a fluke (ie low P val / reject H(0))
    print('ADF Stat is: {}.'.format(dickey_fuller[0]))
    # A lower p val means we can reject the H(0) that our data is NOT stationary
    print('P Val is: {}.'.format(dickey_fuller[1]))
    print('Critical Values (Significance Levels): ')
    for key,val in dickey_fuller[4].items():
        print(key,":",round(val,3))
        
# Differencing the data    
def difference(X):
    diff = X.diff()
    plt.plot(diff)
    plt.show()
    return diff

# X is the series to test
def arima(X, is_stationary = "Y"):
    autocorrelation_plot(X)
    lag = input("What should the lag be in the ARIMA?")
    # If the data is already stationary, no need to difference it (d in pdq)
    # Working with a moving average model (q = 0)
    # Lag value of 1 for the autogression
    model = ARIMA(X, order=(lag,0,0))
    model_fit = model.fit(disp = 0)
    print(model_fit.summary())
    
