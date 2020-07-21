# Importation
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Custom Methods
os.chdir('C:\\Users\\.....')
import custom_methods as cust
import time_series
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
from sklearn.preprocessing import MinMaxScaler

# Data
os.chdir('C:\\Users\\.....')
dat = pd.read_csv('dataset.csv')


# Create a time series DF with date as the index using custom class
features = time_series.create_series(dat, 'v1','date')
# Sort by index
features = features.sort_index()
# Making the RNN dataframe
rnn_df = features.groupby(features.index).sum()
plt.plot(rnn_df)

# Dickey Fuller test of stationarity (custom class)
time_series.stationarity_test(X = rnn_df['gal'])
