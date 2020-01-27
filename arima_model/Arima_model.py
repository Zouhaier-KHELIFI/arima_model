#!/usr/bin/env python
# coding: utf-8

# In[132]:


from matplotlib import pyplot

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier , export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns
import re
import graphviz
import pydotplus
import io
from scipy import misc
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from pandas import datetime
"------------------------------------------------------------------------------------"
def society(timeseries):

    data=pd.read_csv("dataset/"+timeseries+".csv",sep=';',encoding='latin-1',header=0, parse_dates=[0], index_col=0, squeeze=True)
    "------------------------show the type of data-------------------------------------------------"
    #data.describe()
    data.dtypes
    print(data.describe())
    "-----------------------plot data ------------------------------------------"
    df=data['close']
    df.plot()
    pyplot.show()
    
    
    "-----------------------plot data ------------------------------------------"
    
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Add the x-axis and the y-axis to the plot
    ax.plot(data.index.values,
            data['close'], '-o',
            color='purple')
    
    # Set title and labels for axes
    ax.set(xlabel="Date",
           ylabel="close (the last value)",
           title="FB stock")
    # Clean up the x axis dates (reviewed in lesson 4)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
    
    plt.show()
    
    "-----parallel line--------------------------------------------"
    
    rolling_mean = df.rolling(window = 12).mean()
    rolling_std = df.rolling(window = 12).std()
    plt.plot(df, color = 'blue', label = 'Original')
    plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    plt.show()
    
    
    "-----ADF test for verifying the serie is stationary or no --------------------------------------------"
    
    
    result = adfuller(df)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    
    
    "-----------------------ACF data ------------------------------------------"
    
    
    autocorrelation_plot(df)
    pyplot.show()
    "-----------------------PACF data ------------------------------------------"
    
    plot_pacf(df)
    pyplot.show()
    
    
    
    
    
    "-----split data 70/30--------------------------------------------"
    train,test=train_test_split(df,test_size=0.33)
    #dfa=data.ix[:,0]
    #s=data['Ouverture'].astype(float)
    
    "-----------------------fit and --predict----------------------------------------"
    history = [x for x in train]
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    model_fit.plot_predict()
    output = model_fit.forecast()



# In[133]:


society("FBB")





