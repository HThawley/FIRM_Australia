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

os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))

op = pd.read_csv(r"Results\Optimisation_resultx21-consolidated.csv")
op = op[op['Zone'].isna()]
op['event'] = op['event'].fillna(0)

co = pd.read_csv(r"Results/GGTA-consolidated.csv")
co = co[co['Scenario'] == 21] 
co = co[co['n_year'] == -1]
co = co[co['Zone'].isna()]
co['event'] = pd.to_numeric(co['event']).fillna(0)
co.columns = [col.split('.')[0] for col in co.columns]

op = pd.melt(
    op
    , id_vars = ['Scenario', 'Zone', 'n_year', 'event'] 
    , value_vars = ['pv-N_Broken Hill (GW)',
           'pv-N_Central NSW Tablelands (GW)', 'pv-N_Central West NSW (GW)',
           'pv-N_Murray River (GW)', 'pv-N_North West New South Wales (GW)',
           'pv-N_Northern NSW Tablelands (GW)', 'pv-N_Riverland (GW)',
           'pv-Q_Darling Downs (GW)', 'pv-Q_Fitzroy (GW)',
           'pv-S_Eastern Eyre Peninsula (GW)', 'pv-S_Leigh Creek (GW)',
           'pv-S_Northern SA (GW)', 'pv-S_Riverland (GW)', 'pv-S_Roxby Downs (GW)',
           'pv-S_Western Eyre Peninsula (GW)', 'pv-V_Murray River (GW)',
           'wind-N_Broken Hill (GW)', 'wind-N_Central NSW Tablelands (GW)',
           'wind-N_Central West NSW (GW)', 'wind-N_Murray River (GW)',
           'wind-N_North West New South Wales (GW)',
           'wind-N_Northern NSW Tablelands (GW)', 'wind-N_Riverland (GW)',
           'wind-N_Southern NSW Tablelands (GW)', 'wind-Q_Darling Downs (GW)',
           'wind-Q_Fitzroy (GW)', 'wind-S_Eastern Eyre Peninsula (GW)',
           'wind-S_Leigh Creek (GW)', 'wind-S_Mid-North SA (GW)',
           'wind-S_Northern SA (GW)', 'wind-S_Riverland (GW)',
           'wind-S_South East SA (GW)', 'wind-S_Western Eyre Peninsula (GW)',
           'wind-S_Yorke Peninsula (GW)', 'wind-T_King Island (GW)',
           'wind-T_North East Tasmania (GW)', 'wind-T_North West Tasmania (GW)',
           'wind-T_Tasmania Midlands (GW)', 'wind-V_Gippsland (GW)',
           'wind-V_Moyne (GW)', 'wind-V_Murray River (GW)',
           'wind-V_Western Victoria (GW)', 'storage-NSW (GW)', 'storage-QLD (GW)',
           'storage-SA (GW)', 'storage-TAS (GW)', 'storage-VIC (GW)','storage (GWh)' ]
    , var_name = 'ZoneName'
    , value_name = 'capacity'
    )

op['Source'] = op['ZoneName'].str.split('-').apply(lambda x: x[0] if len(x)>=2 else '')
# op['type'] = op['type'].fillna('storage (Energy)')
# op['ZoneName'] = op['ZoneName'].str.split('-').apply(lambda x: x[1] if len(x)==2 else x[0])

op['Source'] = op['Source'].apply(lambda x: {
    'pv':'Solar (GW)'
    , 'wind':'Wind (GW)'
    , 'storage':'Storage (GW)'
    , '':'Storage (GWh)'
    }.get(x))

fig, ax = plt.subplots(figsize = (7, 6))

sns.boxplot(
    op.loc[op.loc[:, 'ZoneName'] != 'storage (GWh)', :]
    , x = 'ZoneName'
    , y = 'capacity'
    # , color = sns.color_palette()[0]
    , hue = 'Source'
    , hue_order = [
        'Solar (GW)'
        , 'Wind (GW)'
        , 'Storage (GW)'
        # , 'Storage (GWh)'
        ]
    , dodge = False
    , width = 0.85
    )
ax.set_yscale('log')
xlim = ax.get_xlim()
# ax.set_xlim(xlim[0], xlim[1]+1)
ax.set_ylabel('Capacity (GW)')
ax.set_xlabel('Zone')
ax.set_xticks([])
ax.set_title('Range of zone capacities')

# fig, ax = plt.subplots(figsize = (6, 8))

# opgrpby = op.groupby(['Source', 'trial']).sum().reset_index()

# sns.barplot(
#     opgrpby.loc[opgrpby.loc[:, 'Source'] != 'Energy (storage)', :]
#     , x = 'Source'
#     , y = 'capacity'
#     # , hue = 'Source'
#     , order = ['Solar (GW)', 'Wind (GW)', 'Storage (GW)', 'Storage (GWh)']
#     # , width = 2.5
#     )
# # ax.legend()
# ax.set_ylabel('Capacity (GW or GWh)')
# ax.set_title('Range of energy mix results')


co = co.rename(columns={
    'LCOE ($/MWh)':'Electricity'
    , 'LCOG ($/MWh)':'Generation'
    , 'LCOB (storage)':'Storage'
    , 'LCOB (transmission)':'Transmission'
    , 'LCOB (spillage & loss)':'Spillage & Loss'
    , 'Pumped Hydro (GW)':'Storage (GW)'
    , 'Pumped Hydro (GWh)':'Storage (GWh)'
    })

co = pd.melt(
    co
    , id_vars = ['Scenario', 'Zone', 'n_year', 'event'] 
    , value_vars = [
        'Solar (GW)', 'Wind (GW)', 'Storage (GW)', 'Storage (GWh)',
        'Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss'
        ]
    , var_name = 'Source'
    , value_name = 'Quantity'
    )

fig, ax = plt.subplots(figsize = (6,4))

comedian = co.groupby(['Scenario','Source']).median().reset_index()

sns.barplot(
    comedian.loc[comedian['Source'].isin(['Solar (GW)', 'Wind (GW)', 'Storage (GW)' , 'Storage (GWh)']),:]
    , x = 'Source'
    , y = 'Quantity'
    , hue = 'Source'
    , dodge = False
    , order = ['Solar (GW)', 'Wind (GW)', 'Storage (GW)','Storage (GWh)']
    , hue_order = ['Solar (GW)', 'Wind (GW)', 'Storage (GW)','Storage (GWh)']
    , ax = ax
    # , markers = 'D'
    )

for a in list(ax.get_children()):
    a.set_zorder(-1)

sns.swarmplot(
    co.loc[co['Source'].isin(['Solar (GW)', 'Wind (GW)', 'Storage (GW)' , 'Storage (GWh)']),:]
    , x = 'Source'
    , y = 'Quantity'
    , hue = 'Source'
    , palette = 'dark:black'
    , order = ['Solar (GW)', 'Wind (GW)', 'Storage (GW)','Storage (GWh)']
    , ax = ax
    , alpha = 0.9
    , dodge = False
    , size = 4
    , zorder = np.inf
    )


ax.legend_.remove()
ax.set_ylabel('Power or Energy (GW or GWh)')
ax.set_title('Range of energy mix')
ax.set_yticks(np.arange(7)*100)
ax.set_xlabel(None)



fig, ax = plt.subplots(figsize = (7,4))

sns.barplot(
    comedian.loc[comedian['Source'].isin(['Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss']),:]
    , x = 'Source'
    , y = 'Quantity'
    , palette = 'Set2'
    , hue = 'Source'
    , dodge = False
    , order = ['Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss']
    , hue_order = ['Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss']
    , ax = ax 
    # , markers = 'D'
    )

for a in list(ax.get_children()):
    a.set_zorder(-1)

sns.swarmplot(
    co.loc[co['Source'].isin(['Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss']),:]
    , x = 'Source'
    , y = 'Quantity'
    , hue = 'Source'
    , palette = 'dark:black'
    , order = ['Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss']
    , ax = ax
    , alpha = 0.9
    , dodge = False
    , size = 4
    , zorder = np.inf
    )
ax.legend_.remove()
ax.set_title('Range of levelised costs')
ax.set_ylabel('Cost ($/MWh)')
ax.set_xlabel(None)


os.chdir('graphs')