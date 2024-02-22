# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:06:25 2024

@author: u6942852
"""
from Setup import scenario, pidx, widx, sidx, bounds

# import numpy as np
from numpy import array
import pandas as pd 
# from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA
from sklearn import cluster
import seaborn as sns 
import matplotlib.pyplot as plt


data = pd.read_csv('Results/OpHist{}.csv'.format(scenario))

varCols=[f'var{n}' for n in range(1,len(data.columns))]

data.columns = ['cost']+varCols

# data = data[data['cost'] != inf]
data = data[data['cost'] < 100]

# model = LinearRegression().fit(data[varCols], data['cost'])

# print(f"""
# score: {model.score(data[varCols], data['cost'])}
# coefs: {model.coef_}
# inter: {model.intercept_}
#       """)

data['solar'] = data[[f'var{n}' for n in range(1, pidx+1)]].sum(axis=1)
data['wind'] = data[[f'var{n}' for n in range(pidx+1, widx+1)]].sum(axis=1)
data['php'] = data[[f'var{n}' for n in range(widx+1, sidx+1)]].sum(axis=1)
data['phs'] = data[f'var{sidx+1}']

data['s/w'] = data['solar']/data['wind']
data['gen'] = data['solar'] + data['wind']
data['phhrs'] = data['phs']/data['php']



brange = array([ub-lb for lb, ub in bounds])
norm_data = data[varCols].div(brange)

data=data.drop(columns=varCols)
#%%
dbc = cluster.DBSCAN(
    eps=0.6
    )
dbc.fit(norm_data)

del norm_data

data['cluster'] = dbc.labels_
data1 = data.loc[data['cluster'] > 0, :]

del dbc

# sns.scatterplot(
#     data=data,
#     x = 's/w',
#     y = 'phhrs', 
#     hue = 'cost',
#     # size='gen'
#     )

plt.figure()
sns.scatterplot(
    data = data1,
    x = 'solar', 
    y = 'wind',
    hue = 'cluster', 
    )

plt.figure()
sns.scatterplot(
    data = data1,
    x = 's/w', 
    y = 'phhrs',
    hue = 'cluster',
    )
plt.figure()
sns.scatterplot(
    data = data1,
    x = 's/w', 
    y = 'gen',
    hue = 'cluster',
    )

