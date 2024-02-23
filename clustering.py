# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:06:25 2024

@author: u6942852
"""
from Setup import *

import numpy as np
import pandas as pd 
# from sklearn.linear_model import LinearRegression
# from sklearn.decomposition import PCA
from sklearn import cluster
import seaborn as sns 
import matplotlib.pyplot as plt



if args.x > 2: 
    data = pd.read_csv('Results/OpHist{}-{}.csv'.format(scenario, args.x))
else: 
    data = pd.read_csv('Results/OpHist{}.csv'.format(scenario))

varCols=[f'var{n}' for n in range(1,len(data.columns))]

data.columns = ['cost']+varCols

# data = data[data['cost'] != inf]
data = data[data['cost'] < costConstraint]
print("full data length: ", len(data))
# data = data.iloc[:len(data)//10,:]
print("trunced data length: ", len(data))

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



brange = np.array([ub-lb for lb, ub in bounds])
# norm_data = data[varCols].div(brange)

# data=data.drop(columns=varCols)
#%%
# dbc = cluster.DBSCAN(
#     eps=0.1
#     )
# dbc.fit(norm_data)

# del norm_data

# data['cluster'] = dbc.labels_
# data1 = data.loc[data['cluster'] > 0, :]
data1 = data.loc[:, :]

# del dbc

# sns.scatterplot(
#     data=data,
#     x = 's/w',
#     y = 'phhrs', 
#     hue = 'cost',
#     # size='gen'
#     )

# plt.figure()
# sns.scatterplot(
#     data = data1,
#     x = 'solar', 
#     y = 'wind',
#     hue = 'cost', 
#     )

# plt.figure()
# sns.scatterplot(
#     data = data1,
#     x = 's/w', 
#     y = 'phhrs',
#     hue = 'cost',
#     )
# plt.figure()
# sns.scatterplot(
#     data = data1,
#     x = 's/w', 
#     y = 'gen',
#     hue = 'cluster',
#     )
# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='s/w',import bokeh
#     y='cost',
#     hue='gen'
# )
# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='php',
#     y='phs',
#     hue='cost'
# )

# plt.show()


from matplotlib import cm, colors
cmap = cm.RdYlGn
norm = colors.Normalize(vmin=-1, vmax=1)
cb = cm.ScalarMappable(norm=norm, cmap=cmap)

data2=data1[['cost']+varCols].corr()
for i in range(len(data2.columns)):
    data2.iloc[i,i] = np.nan
p = plt.matshow(data2, cmap=cmap, norm=norm)

plt.colorbar(cb)


# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='var4',
#     y='solar',
#     hue='cost',
# )

# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='var4',
#     y='wind',
#     hue='cost',
# )

# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='var4',
#     y='php',
#     hue='cost',
# )

# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='var4',
#     y='phs',
#     hue='cost',
# )

# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='var4',
#     y='var5',
#     hue='cost',
# )

plt.figure()
data['var4'].hist()

plt.show()


