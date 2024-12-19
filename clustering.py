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
#%%
size = 0.5
costConstraint = 1.1

file = r'C:/Users/u6942852/Desktop/History11.csv'
file =r"C:\Users\u6942852\Documents\Repos\FIRMPlus_Australia\Results\History12-children.csv"
data = pd.read_csv(file, header=None)

# if args.x > 2: 
#     data = pd.read_csv('Results/OpHist{}-{}.csv'.format(scenario, args.x))
# else: 
#     data = pd.read_csv('Results/OpHist{}.csv'.format(scenario))

varCols=[f'var{n}' for n in range(1, pzones+wzones+nodes+2)]

# data.columns = ['cost']+varCols
data.columns = ['cost', 'gen', 'cuts']+varCols

mincost = data['cost'].min()

resolved = data['cuts'].max()
data = data.loc[data['cuts'] == resolved,:]
data=data.sort_values('cost', ascending=False)
data = data.drop(columns=['gen', 'cuts'])
# costConstraint = data['cost'].min()*1.5

print("full data length: ", len(data))
data = data[data['cost'] != np.inf]
data = data[data['cost'] < costConstraint*mincost]
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

# dbc = cluster.DBSCAN(
#     eps=0.1
#     )
# dbc.fit(norm_data)

# del norm_data

# data['cluster'] = dbc.labels_
# data1 = data.loc[data['cluster'] > 0, :]
data1 = data.loc[:, :]
data1=data1.round(5)
# del dbc

# sns.scatterplot(
#     data=data,
#     x = 's/w',
#     y = 'phhrs', 
#     hue = 'cost',
#     size='gen'
#     )


# plt.figure()
# sns.scatterplot(
#     data = data1,
#     x = 'solar', 
#     y = 'wind',
#     hue = 'cost', 
#     size = size,
#     )

# plt.figure()
# sns.pairplot(
#     data=data1,
#     vars=['cost','solar', 'wind', 'php', 'phs',
#            's/w', 'gen', 'phhrs']
#     )

# plt.figure()
# sns.scatterplot(
#     data = data1,
#     x = 's/w', 
#     y = 'phhrs',
#     hue = 'cost',
#     # size = size,
#     )
# # plt.figure()
# # sns.scatterplot(
# #     data = data1,
# #     x = 's/w', 
# #     y = 'gen',
# #     hue = 'cost',
# #     # size = size,
# #     )
# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='s/w',
#     y='gen',
#     hue='cost',
#     # size = size,
# )
# plt.figure()
# sns.scatterplot(
#     data=data1,
#     x='php',
#     y='phs',
#     hue='cost',
#     # size = size,
# )

# plt.figure()
# sns.scatterplot(
#     data=data1, 
#     x='gen',
#     y='php',
#     hue='cost'
#     )

# plt.figure()
# sns.scatterplot(
#     data=data1, 
#     x='gen',
#     y='phs',
#     hue='cost'
#     )

# plt.figure()
# sns.scatterplot(
#     data=data1, 
#     x='gen',
#     y='phhrs',
#     hue='cost'
#     )


# plt.figure()
# sns.histplot(
#     data=data1, 
#     x='gen',
#     y='php',
#     bins = (data1['gen'].round(4).nunique(), 
#             data1['php'].round(4).nunique())
#     )

# plt.figure()
# sns.scatterplot(
#     data=data1.groupby(['gen', 'php'])['cost'].min().reset_index(),
#     x='gen', 
#     y='php',
#     hue='cost',    
#     )

def continuous_heatmap(data, x, y, val, agg='min', colormap='rocket', reverse_color=False, ax=None, 
                        fig=None, x_bins='max', y_bins='max'):
    assert agg in ('min', 'max', 'count', 'mean','median')
    ax = plt.gca() if ax is None else ax
    fig = plt.gcf() if fig is None else fig
    
    colormap = colormap+'_r' if reverse_color else colormap

    xmin, xmax = data[x].min(), data[x].max()
    ymin, ymax = data[y].min(), data[y].max()
    
    x_bins = data[x].nunique() if x_bins=='max' else x_bins
    y_bins = data[y].nunique() if y_bins=='max' else y_bins

    if agg == 'min': data=data.groupby([x, y])[val].min()
    if agg == 'max': data=data.groupby([x, y])[val].max()
    if agg == 'count': data=data.groupby([x, y])[val].count()
    if agg == 'mean': data=data.groupby([x, y])[val].mean()
    if agg == 'median': data=data.groupby([x, y])[val].median()
    
    n_levels = min(data.nunique(), 1000)
    z = data.reset_index()\
        .pivot(index=y, columns=x, values=val).to_numpy()
    
    x, y = np.meshgrid(
        np.linspace(xmin, xmax, x_bins),
        np.linspace(ymin, ymax, y_bins))
    
    c = ax.contourf(x,y,z, n_levels, cmap=colormap, corner_mask=False)
    fig.colorbar(c, ax=ax)
    
    
fig, ax = plt.subplots()
continuous_heatmap(data1, 'gen', 'php', 'cost', 'min', reverse_color=True, ax=ax, fig=fig)
ax.set_ylim([0.015, 0.175])
ax.set_xlim([0.015, 0.175])
fig, ax = plt.subplots()
continuous_heatmap(data1, 'gen', 'php', 'cost', 'count', ax=ax, fig=fig)
ax.set_ylim([0.015, 0.175])
ax.set_xlim([0.015, 0.175])

fig, ax = plt.subplots()
continuous_heatmap(data1, 'solar', 'wind', 'cost', 'min', reverse_color=True, ax=ax, fig=fig)
ax.set_ylim([0.015, 0.175])
ax.set_xlim([0.015, 0.175])
fig, ax = plt.subplots()
continuous_heatmap(data1, 'solar', 'wind', 'cost', 'count', ax=ax, fig=fig)
ax.set_ylim([0.015, 0.175])
ax.set_xlim([0.015, 0.175])
            
plt.show()

