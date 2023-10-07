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

def lambdaZoneNames(scenario, stormZone=None, wdir=None):
    if pd.isna(stormZone): return None
    
    pidx, widx, sidx, headers, stormZoneIndx = zoneTypeIndx(scenario, stormZone, wdir)
    zoneNames = [name[7:-5] for name in np.array(headers[pidx:widx])[stormZoneIndx]]
    return zoneNames

def readPrintedArray(txt):      
    if txt == 'None': return None
    txt = re.sub(r"(?<!\[)\s+(?!\])", r",", txt)
    return np.array(literal_eval(txt), dtype =int)

def zoneTypeIndx(scenario, stormZone=None, wdir=None):
    if wdir is not None: 
        active_dir = os.getcwd()
        os.chdir(wdir)
    Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
    PVl   = np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
    Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)
    names = np.char.replace(np.genfromtxt('Data\ZoneDict.csv', delimiter=',', dtype=str), 'ï»¿', '')
    if isinstance(stormZone, str):        
        if str(stormZone) == 'all': stormZone = ...
        else: stormZone = readPrintedArray(stormZone) 

    if scenario<=17:
        node = Nodel[scenario % 10]
        names = names [np.where(np.append(PVl, Windl)==node)[0]] 
        pzones = len(np.where(PVl==node)[0])
        wzones = len(np.where(Windl==node)[0])
        coverage = np.array([node])
        
        zones = np.zeros(Windl.shape)
        zones[stormZone] = 1 
        zones = zones[np.where(Windl==node)[0]]
        stormZoneIndx = np.where(zones==1)[0]
        
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
        zones[stormZone] = 1 
        zones = zones[np.where(np.in1d(Windl, coverage)==True)[0]]
        stormZoneIndx = np.where(zones==1)[0]
        
        names = names[np.where(np.in1d(np.append(PVl, Windl), coverage)==True)[0]]
        pzones = len(np.where(np.in1d(PVl, coverage)==True)[0])
        wzones = len(np.where(np.in1d(Windl, coverage)==True)[0])
       
    nodes = len(coverage)
    pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)
    
    headers = (['pv-'   + name + ' (GW)' for name in names[:pidx]] + 
               ['wind-' + name + ' (GW)' for name in names[pidx:widx]] + 
               ['storage-' + name + ' (GW)' for name in coverage] + 
               ['storage (GWh)'])
    
    if stormZone is None: stormZoneIndx=None
    
    if wdir is not None: 
        os.chdir(active_dir)
    return pidx, widx, sidx, headers, stormZoneIndx

def lambdaReadDeficit(scenario, zone, n_year):
    try: return pd.read_csv(fr'S{scenario}-{zone}-{n_year}-{event}.csv')['StormDeficit'].sum()
    except FileNotFoundError: 
        if n_year == -1: return 0
        else: return np.nan


#%%
if __name__ == '__main__':
    
    event = 's' #storm
    event = 'd' #drought
    
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
    
    
    comparisonColumns = ['Solar (GW)', 'Wind (GW)', 'Hydro & Bio (GW)', 'Pumped Hydro (GW)', 
                         'Pumped Hydro (GWh)', 'LCOE ($/MWh)', 'LCOG ($/MWh)', 
                         'LCOB (storage)', 'LCOB (transmission)', 'LCOB (spillage & loss)']
    
    baseCaps = data.loc[data['n_year'] == -1, ['Scenario']+comparisonColumns]
    baseCaps['Total Gen'] = baseCaps[['Solar (GW)', 'Wind (GW)', 'Hydro & Bio (GW)']].sum(axis=1)
    baseCaps = baseCaps
    
    data = pd.merge(data, baseCaps, on = 'Scenario', how = 'inner', suffixes = ('', '_relative'))
    
    relCols = [col + '_relative' for col in comparisonColumns]
    
    data[relCols] = -data[relCols].subtract(data[comparisonColumns].values, axis = 0)
    
    data['GenerationAffected'] = 100*data['zone capacity'] / data['Total Gen']
    
    data['Energy Deficit (TWh p.a.)'] = data[['Scenario', 'Zone', 'n_year']].apply(lambda row: lambdaReadDeficit(*row)*0.5/10/1e6, axis = 1)
    
    
    # data = pd.read_excel(r'Graph1DummyData.xlsx')
    data['deficit%'] = data['Energy Deficit (TWh p.a.)'] / data['Energy Demand (TWh p.a.)']
    
    
    # fig, ax = plt.subplots(figsize=(8, 4))
    # sns.scatterplot(data, 
    #                 x = 'GenerationAffected',
    #                 y = 'LCOE ($/MWh)_relative',
    #                 hue = 'Grid', 
    #                 size = 'Energy Demand (TWh p.a.)')
    
    # ax.set_ylabel('Cost Ratio (%)')
    # ax.set_xlabel('Generation affected (%)')
    # ax.set_xlim([None, 50])
    # ax.set_xticks(np.arange(6)/0.2)
    # plt.legend()
    # plt.show()
    
    # fig, ax = plt.subplots(figsize=(8, 4))
    # sns.scatterplot(data, 
    #                 x = 'GenerationAffected',
    #                 y = 'deficit%', 
    #                 hue = 'Grid', 
    #                 size = 'LCOE ($/MWh)_relative')
    
    # ax.set_ylabel('Energy Deficit (%)')
    # ax.set_xlabel('Generation affected (%)')
    # ax.set_xlim([None, 50])
    # ax.set_xticks(np.arange(6)/0.2)
    # plt.legend()
    # plt.show()
    # plt.show()
    
    data['GridZone'] = data['Grid'] + data['Zone']
    data1 = pd.melt(data
                    , id_vars = ['GridZone','Scenario', 'Zone Name', 'Zone', 'GenerationAffected']
                    , value_vars = ['LCOE ($/MWh)_relative', 'LCOG ($/MWh)_relative', 
                                    'LCOB (storage)_relative', 'LCOB (transmission)_relative', 
                                    'LCOB (spillage & loss)_relative'])
    data1['variable'] = data1['variable'].str.replace('_relative', '')
    data1['GenerationAffected'] = data1['GenerationAffected'].round(1)
    
    fig, ax = plt.subplots(figsize = (12,4))
    sns.barplot(data1#.loc[data1.loc[:,'Scenario'] == 21, :]
                , x = 'GridZone'
                , y = 'value'
                , hue = 'variable'
                )
    ax.axhline(0.0, linewidth = 0.5, color='black')
    
    # ax.set_xlim([None, 6])
    ax.set_xlabel('Zone affected')
    ax.set_ylabel('Cost of resilient grid relative to ordinary')
    ax.set_title('Relative costs by Storm Location')
    plt.legend()
    plt.show()
    
    
    data1 = pd.melt(data
                    , id_vars = ['GridZone', 'Scenario', 'Zone Name', 'Zone', 'GenerationAffected']
                    , value_vars = ['Solar (GW)_relative'
                                    , 'Wind (GW)_relative'
                                    # , 'Hydro & Bio (GW)_relative'
                                    , 'Pumped Hydro (GW)_relative'
                                    # , 'Pumped Hydro (GWh)_relative'
                                    ])
    data1['variable'] = data1['variable'].str.replace('_relative', '')
    data1['GenerationAffected'] = data1['GenerationAffected'].round(1)
    
    fig, ax = plt.subplots(figsize = (12,4))
    sns.barplot(data1#.loc[data1.loc[:,'Scenario'] == 21, :]
                , x = 'GridZone'
                , y = 'value'
                , hue = 'variable'
                )
    ax.axhline(0.0, linewidth = 0.5, color='black')
    
    # ax.set_xlim([None, 6])
    ax.set_xlabel('Zone affected')
    ax.set_ylabel('Change to generation (GW)')
    ax.set_title('Relative Energy Mix by Location')
    plt.legend()
    plt.show()
    
    
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
    os.chdir('graphs')
    
    
    
