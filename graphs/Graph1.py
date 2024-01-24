# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:43:02 2023

@author: hmtha
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
# from matplotlib import cm
import os 
from ast import literal_eval
import re

# colormap = cm.inferno

#%%

def lambdaZoneNames(scenario, eventZone=None, wdir=None):
    if pd.isna(eventZone): return None
    
    pidx, widx, sidx, headers, eventZoneIndx = zoneTypeIndx(scenario, eventZone, wdir)
    zoneNames = [name[7:-5] for name in np.array(headers[pidx:widx])[eventZoneIndx]]
    return zoneNames

def readPrintedArray(txt):      
    if txt == 'None': return None
    txt = re.sub(r"(?<!\[)\s+(?!\])", r",", txt)
    return np.array(literal_eval(txt), dtype =int)

def zoneTypeIndx(scenario, eventZone=None, wdir=None):
    if wdir is not None: 
        active_dir = os.getcwd()
        os.chdir(wdir)
    Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
    PVl   = np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
    Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)
    names = np.char.replace(np.genfromtxt('Data\ZoneDict.csv', delimiter=',', dtype=str), 'ï»¿', '')
    if isinstance(eventZone, str):        
        if str(eventZone) == 'all': eventZone = ...
        else: eventZone = readPrintedArray(eventZone) 

    if scenario<=17:
        node = Nodel[scenario % 10]
        names = names [np.where(np.append(PVl, Windl)==node)[0]] 
        pzones = len(np.where(PVl==node)[0])
        wzones = len(np.where(Windl==node)[0])
        coverage = np.array([node])
        
        zones = np.zeros(Windl.shape)
        zones[eventZone] = 1 
        zones = zones[np.where(Windl==node)[0]]
        eventZoneIndx = np.where(zones==1)[0]
        
    if scenario>=21:
        coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1]
        
        zones = np.zeros(Windl.shape)
        zones[eventZone] = 1 
        zones = zones[np.where(np.in1d(Windl, coverage)==True)[0]]
        eventZoneIndx = np.where(zones==1)[0]
        
        names = names[np.where(np.in1d(np.append(PVl, Windl), coverage)==True)[0]]
        pzones = len(np.where(np.in1d(PVl, coverage)==True)[0])
        wzones = len(np.where(np.in1d(Windl, coverage)==True)[0])
       
    nodes = len(coverage)
    pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)
    
    headers = (['pv-'   + name + ' (GW)' for name in names[:pidx]] + 
               ['wind-' + name + ' (GW)' for name in names[pidx:widx]] + 
               ['storage-' + name + ' (GW)' for name in coverage] + 
               ['storage (GWh)'])
    
    if eventZone is None: eventZoneIndx=None
    
    if wdir is not None: 
        os.chdir(active_dir)
    return pidx, widx, sidx, headers, eventZoneIndx

def lambdaReadDeficit(scenario, zone, n_year, event):
    try: return pd.read_csv(fr'S{scenario}-{zone}-{n_year}-{event}.csv')['eventDeficit'].sum()
    except FileNotFoundError: 
        if n_year == -1: return 0
        else: return np.nan

    
    
#%%
if __name__ == '__main__':
    
    events = ('e',)
    
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
    os.chdir('Results')
    
    
    
    data = pd.read_csv('GGTA-consolidated.csv')
    # data = pd.read_csv(r'C:\Users\hmtha\Desktop\GGTA-consolidated.csv')
    data['Grid'] = data['Scenario'].map({11:'NSW', 
                                         # 12:'NT',
                                         13:'QLD',
                                         14:'SA', 
                                         15:'TAS', 
                                         16:'VIC', 
                                         # 17:'WA', 
                                         21:'NEM'})
    data = data.dropna(subset = ['Grid']).reset_index(drop=True)
    
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
    data['Zone Name'] = data[['Scenario', 'Zone']].apply(lambda x: lambdaZoneNames(*x), axis=1)
    os.chdir('Results')
    
    
    genColumns = ['Solar (GW)', 'Wind (GW)', 'Hydro & Bio (GW)', 'Pumped Hydro (GW)', 
                         'Pumped Hydro (GWh)']
    costColumns = ['LCOE ($/MWh)', 'LCOG ($/MWh)', 'LCOB (storage)', 'LCOB (transmission)', 
                   'LCOB (spillage & loss)']
    
    baseCaps = data.loc[data['n_year'] == -1, ['Scenario']+genColumns+costColumns]
    baseCaps['Total Gen'] = baseCaps[['Solar (GW)', 'Wind (GW)', 'Hydro & Bio (GW)']].sum(axis=1)
    
    minbaseCaps = baseCaps.groupby(['Scenario']).min()
    maxbaseCaps = baseCaps.groupby(['Scenario']).max()
    meanbaseCaps = baseCaps.groupby(['Scenario']).mean()
    
    precision = ((maxbaseCaps - minbaseCaps)/meanbaseCaps/2)
    
    minCosts = baseCaps.groupby(['Scenario'])['LCOE ($/MWh)'].min().reset_index()
    baseCaps = pd.merge(baseCaps, minCosts, on = ['Scenario', 'LCOE ($/MWh)'], how='inner')
    baseCaps = baseCaps[~baseCaps.duplicated()]
    
    x, y = baseCaps.groupby('Scenario'), baseCaps.groupby('Scenario')
    x = baseCaps.groupby('Scenario').max().multiply(1+precision).reset_index()
    y = baseCaps.groupby('Scenario').max().multiply(1-precision).reset_index()
    baseCaps = pd.concat([baseCaps, x, y], ignore_index = True)
    
    
    
    data = pd.merge(data, baseCaps, on = 'Scenario', how = 'inner', suffixes = ('', '_relative'))
    
    relgenCols = [col + '_relative' for col in genColumns]
    relcostCols = [col + '_relative' for col in costColumns]
    
    data[relcostCols] = -data[relcostCols].subtract(data[costColumns].values, axis = 0)
    data[relgenCols] = 100-100*data[relgenCols].divide(data[genColumns].values, axis=0)
    
    data['GenerationAffected'] = 100*data['zone capacity'] / data['Total Gen']
    
    data['Energy Deficit (TWh p.a.)'] = data[['Scenario', 'Zone', 'n_year', 'event']].apply(lambda row: lambdaReadDeficit(*row)*0.5/10/1e6, axis = 1)
    
    
    data['deficit%'] = data['Energy Deficit (TWh p.a.)'] / data['Energy Demand (TWh p.a.)']
    
    
    data = data[data['event'].isin(events)]
    data = data[data['Scenario'] == 21]
    data['GridZone'] = data['Grid'] + data['Zone'] + data['event']
    data1 = pd.melt(data
                    , id_vars = ['GridZone','Scenario', 'Zone Name', 'Zone', 'GenerationAffected']
                    , value_vars = ['LCOE ($/MWh)_relative', 'LCOG ($/MWh)_relative', 
                                    'LCOB (storage)_relative', 'LCOB (transmission)_relative', 
                                    'LCOB (spillage & loss)_relative'])
    data1['variable'] = data1['variable'].str.replace('_relative', '')
    data1['variable'] = data1['variable'].apply(lambda x: {
        'LCOE ($/MWh)':'Electricity'
        , 'LCOG ($/MWh)':'Generation'
        , 'LCOB (storage)':'Storage'
        , 'LCOB (transmission)':'Transmission'
        , 'LCOB (spillage & loss)':'Spillage & Loss'
        }.get(x,x))
    data1['GenerationAffected'] = data1['GenerationAffected'].round(1)
    data1['Zone Name'] = data1['Zone Name'].apply(lambda x: x[0])
    data1['Zone Name'] = data1['Zone Name'].apply(lambda x: {
        'Fitzroy':'Fitzroy (QLD)'
        , 'Leigh Creek':'Leigh Creek (SA)'
        }.get(x,x))
    
    #%%
    
    # fig, axs = plt.subplots(2, figsize = (11,6))
    # plt.subplots_adjust(hspace = 0.4)
    # ax = axs[1]
    
    fig, ax = plt.subplots(figsize=(8,6), dpi=2000)
    
    sns.barplot(
        data1#.loc[data1.loc[:,'Scenario'] == 21, :]
        , x = 'Zone Name'
        , y = 'value'
        , hue = 'variable'
        , hue_order = ['Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss']
        , palette = 'Set2'
        , order = ['Fitzroy (QLD)', 'Southern NSW Tablelands', 'Leigh Creek (SA)']
        , ax = ax
        )
    ax.axhline(0.0, linewidth = 0.5, color='black')
    # legend_.remove()

    ax.grid(True, 'major', 'y')
    # ax.set_xlim([None, 7.5])
    # ax.set_ylim([-0.501,1.501])
    # ax.set_yticks(np.arange(-0.5, 1.75, 0.5, float))
    ax.set_xlabel('Zone affected by HILP wind events')
    ax.set_ylabel('Relative Cost ($/MWh)')
    # ax.set_title('b) Levelised costs of resilient grids relative to base scenario')
    ax.set_title('Levelised costs of resilient grids relative to base scenario')

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.95, pos.height])
    
    lns, labs = ax.get_legend_handles_labels()

    
    ax.legend(
        lns 
        , labs 
        , loc = 'center right'
        , bbox_to_anchor = (1.285, 0.5)
        ) 
    # plt.show()
    
#%%    
    data2 = pd.melt(
        data
        , id_vars = ['GridZone', 'Scenario', 'Zone Name', 'Zone', 'GenerationAffected']
        , value_vars = ['Solar (GW)_relative'
                        , 'Wind (GW)_relative'
                        # , 'Hydro & Bio (GW)_relative'
                        , 'Pumped Hydro (GW)_relative'
                        , 'Pumped Hydro (GWh)_relative'
                        ])
    
    data2['variable'] = data2['variable'].str.replace('_relative', '')
    data2['GenerationAffected'] = data2['GenerationAffected'].round(1)
    data2['Zone Name'] = data2['Zone Name'].apply(lambda x: x[0])
    data2['Zone Name'] = data2['Zone Name'].apply(lambda x: {
        'Fitzroy':'Fitzroy (QLD)'
        , 'Leigh Creek':'Leigh Creek (SA)'
        }.get(x,x))
    data2['variable'] = data2['variable'].apply(lambda x: {
        'Pumped Hydro (GW)':'Storage (GW)', 
        'Pumped Hydro (GWh)': 'Storage (GWh)'}.get(x,x))
    
    fig, ax1 = plt.subplots(figsize = (8,6), dpi=2000)
    # ax1 = axs[0]

    sns.barplot(
        data2#.loc[data2.loc[:,'variable'] != 'Pumped Hydro (GWh)', :]
        , x = 'Zone Name'
        , y = 'value'
        , hue = 'variable'
        , hue_order = ['Solar (GW)', 'Wind (GW)', 'Storage (GW)', 'Storage (GWh)']
        , order = ['Fitzroy (QLD)', 'Southern NSW Tablelands', 'Leigh Creek (SA)']
        , ax = ax1
        )
    ax1.axhline(0.0, linewidth = 0.5, color='black')

    # ax2 = ax1.twinx()
    # s = ax2.stem(
    #     data2.loc[data2.loc[:,'variable'] == 'Storage (GWh)', :]['Zone']
    #     , data2.loc[data2.loc[:,'variable'] == 'Storage (GWh)', :]['value']
    #     , label = 'Storage (GWh)'
    #     , basefmt=' '
    #     , linefmt='r'
    #     , markerfmt='r'
    #     # , ax = ax2
    #     # , color = sns.color_palette()[3]
    #     )
    # s[0].set_color(sns.color_palette()[3])
    # s[1].set_color(sns.color_palette()[3])
    # s[2].set_color(sns.color_palette()[3])

    
    ax1.grid(True, 'major', 'y')
    # ax1.set_xlim([None, 6.5])
    # yscaleFactor = 1+int(ax1.get_ylim()[1]/ax2.get_ylim()[1])
    # ax1.set_ylim([-4.05,6.05])
    # ax1.set_yticks(np.arange(-4.0, 8.0, 2.0, float))

    # ax2.set_ylim([25*val for val in ax1.get_ylim()])
    # ax2.set_yticks([25*val for val in ax1.get_yticks()])
    
    # ax1.set_yticks(np.arange(-2,5)*5.0)    
    ax1.set_xlabel('Zone affected by HILP wind events')
    ax1.set_ylabel('Relative Capacity (%)')
    # ax2.set_ylabel('Relative Energy Storage Capacity (GWh)')
    # ax1.set_title('a) Energy mix of resilient grids relative to base case')
    ax1.set_title('Energy resources of resilient grids relative to base case')

    pos = ax1.get_position()
    ax1.set_position([pos.x0, pos.y0, pos.width*0.95, pos.height])
    
    lns, labs = ax1.get_legend_handles_labels()
    # lns2, labs2 = ax2.get_legend_handles_labels()
    
    ax1.legend(
        lns #+ lns2
        , labs #+ labs2
        , loc = 'center right'
        , bbox_to_anchor = (1.285, 0.5)
        ) 

    plt.show()
    
#%%    
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
    os.chdir('graphs')
    
    
    
