# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 00:17:40 2023

@author: hmtha
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 
import seaborn as sns
import numpy as np
import os 

from graphutils import directory_up, standardiseNas, adjust_legend

directory_up()

scenario = 21
separate_plots = False
dpi = 200

#%%
# =============================================================================
# Data preprocessing
# =============================================================================
def read_and_filter_df(path):
    df = pd.read_csv(path)
    df = df.applymap(standardiseNas)
    df = df[df['Scenario'] == scenario] 
    df = df[df['Zone'].isna()]
    df = df.dropna(subset=['event'])
    df = df
    return df.reset_index(drop=True).drop(index=0).reset_index(drop=True)
    
op = read_and_filter_df(fr"Results\Optimisation_resultx{scenario}-consolidated.csv")
co = read_and_filter_df(r"Results/GGTA-consolidated.csv")

op = pd.melt(
    op,
    id_vars = ['Scenario', 'Zone', 'n_year', 'event'],
    value_vars = [col for col in op.columns if 'GW' in col],
    var_name = 'ZoneName',
    value_name = 'capacity',
    )

op['Source'] = op['ZoneName'].str.split('-').apply(lambda x: x[0] if len(x)>=2 else '')

op['Source'] = op['Source'].apply(lambda x: {
    'pv':'Solar (GW)',
    'wind':'Wind (GW)',
    'storage':'Storage (GW)',
    '':'Storage (GWh)',
    }.get(x))

co = co.rename(columns={
    'LCOE ($/MWh)':'Electricity',
    'LCOG ($/MWh)':'Generation',
    'LCOB (storage)':'Storage',
    'LCOB (transmission)':'Transmission',
    'LCOB (spillage & loss)':'Spillage & Loss',
    'Pumped Hydro (GW)':'Storage (GW)',
    'Pumped Hydro (GWh)':'Storage (GWh)',
    })

co = pd.melt(
    co,
    id_vars = ['Scenario', 'Zone', 'n_year', 'event'] ,
    value_vars = [
        'Solar (GW)', 'Wind (GW)', 'Storage (GW)', 'Storage (GWh)', 'Electricity', 
        'Generation', 'Storage', 'Transmission', 'Spillage & Loss'],
    var_name = 'Source',
    value_name = 'Quantity',
    )

comedian = co.groupby(['Scenario','Source']).median().reset_index()

#%%
# =============================================================================
# Boxplot of capacities at each zone
# =============================================================================
fig, ax = plt.subplots(figsize = (8, 6), dpi=dpi)

sns.boxplot(
    data = op.loc[op.loc[:, 'ZoneName'] != 'storage (GWh)', :],
    x = 'ZoneName',
    y = 'capacity',
    hue = 'Source',
    hue_order = ['Solar (GW)', 'Wind (GW)','Storage (GW)',],
    dodge = False,
    width = 0.85,
    ax = ax,
    )

ax.set_yscale('log')
ax.set_ylabel('Capacity (GW)')
ax.set_xlabel('Zone')
ax.set_xticks([])
ax.set_title('Range of zone capacities')
ax.legend_.remove()
adjust_legend(ax, 1.28, 0.5, loc = 'center right')

#%%

def barplot(ax, sources, order, palette='tab10'):
    sns.barplot(
        data = comedian.loc[comedian['Source'].isin(sources),:],
        x = 'Source',
        y = 'Quantity',
        hue = 'Source',
        palette = palette,
        dodge = False,
        order = order,
        hue_order = order,
        ax = ax,
        )
    
def swarmplot(ax ,sources, order, palette='dark:black'):
    sns.swarmplot(
        data = co.loc[co['Source'].isin(sources),:],
        x = 'Source',
        y = 'Quantity',
        hue = 'Source',
        palette = 'dark:black',
        order = order,
        ax = ax,
        alpha = 0.9,
        dodge = False,
        size = 4,
        zorder = np.inf,
        )

#%%
# =============================================================================
# costs and energy resources plots
# =============================================================================
if separate_plots is True: 
    axs = []
    size = (8,3)
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    axs.append(ax)
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    axs.append(ax)
    prefix = ['','']
else: 
    fig, axs = plt.subplots(2, figsize = (8,6), dpi = dpi)
    fig.subplots_adjust(hspace=0.4)
    prefix = ['a) ', 'b) ']

# Resources plot
axs[0] = [axs[0], axs[0].twinx()]
order = ['Solar (GW)', 'Wind (GW)', 'Storage (GW)','Storage (GWh)']
sources = [['Solar (GW)', 'Wind (GW)', 'Storage (GW)'], ['Storage (GWh)']]

for i in range(2):
    barplot(axs[0][i], sources[i], order)
    swarmplot(axs[0][i], sources[i], order)
    axs[0][i].legend_.remove()

axs[0][0].set_ylabel('Power (GW)')
axs[0][1].set_ylabel('Energy (GWh)')
axs[0][0].set_title(prefix[0] + 'Range of energy mix')
# axs[0][1].set_yticks(np.arange(7)*100)
axs[0][0].set_xlabel(None)
# axs[0][0].set_ylim([0,100])
# axs[0][1].set_ylim([0,600])


# Costs plot
order = ['Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss']

barplot(axs[1], order, order, 'Set2')
swarmplot(axs[1], order, order)

axs[1].legend_.remove()
axs[1].set_title(prefix[1] + 'Range of levelised costs')
axs[1].set_ylabel('Cost ($/MWh)')
axs[1].set_xlabel(None)


os.chdir('graphs')