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

from graphutils import zoneTypeIndx, directory_up, adjust_legend



#%%

def lambdaZoneNames(scenario, eventZone=None, wdir=None):
    if pd.isna(eventZone): return None
    
    pidx, widx, sidx, headers, eventZoneIndx = zoneTypeIndx(scenario, eventZone, wdir)
    zoneNames = [name[7:-5] for name in np.array(headers[pidx:widx])[eventZoneIndx]]
    return zoneNames



    
    
#%%
if __name__ == '__main__':
    
# =============================================================================
#   analysis parameters
# =============================================================================
    events = ('e',)
    scenario = 21
    # plot as separate subplots
    separate_plots = True
    # figure res
    dpi = 100
# =============================================================================
#     directory management
# =============================================================================
    directory_up()
#%%
# =============================================================================
#     read in and pre-process data   
# =============================================================================
    data = pd.read_csv('Results/GGTA-consolidated.csv')
    # add grid name column
    data['Grid'] = data['Scenario'].map({11:'NSW', 
                                         # 12:'NT',
                                         13:'QLD',
                                         14:'SA', 
                                         15:'TAS', 
                                         16:'VIC', 
                                         # 17:'WA', 
                                         21:'NEM'})
    # remove irrelevant rows
    data = data.dropna(subset = ['Grid']).reset_index(drop=True)
    # clean up nulls
    data = data.applymap(lambda e: {'None':pd.NA}.get(e,e))
    
    # Add more informative zone name for results display
    data['Zone Name'] = data[['Scenario', 'Zone']].apply(lambda x: lambdaZoneNames(*x), axis=1)
    
    
    # os.chdir('Results')
    
# =============================================================================
# Calculate baseline values and precision bars
# =============================================================================
    # More convenient lists of relevant columns
    gens = ['Solar (GW)', 'Wind (GW)', 'Pumped Hydro (GW)', 'Pumped Hydro (GWh)',
            # 'Hydro & Bio (GW)', Hydro & Bio capacity is not variable, only usage
             ]
    costs = ['LCOE ($/MWh)', 'LCOG ($/MWh)', 'LCOB (storage)', 'LCOB (transmission)', 
                   'LCOB (spillage & loss)']
    
    # Filter rows and columns to find population of cost-optimisation reruns
    baseCaps = data.loc[data['Zone'].isna(), ['Scenario']+gens+costs+['Hydro & Bio (GW)']]
    
    # add total generation column
    baseCaps['Total Gen'] = baseCaps[['Solar (GW)', 'Wind (GW)', 'Hydro & Bio (GW)']].sum(axis=1)
    
    # caluclate mins, maxs, and means of variables of optimisation variables
    minbaseCaps = baseCaps.groupby(['Scenario']).min()
    maxbaseCaps = baseCaps.groupby(['Scenario']).max()
    meanbaseCaps = baseCaps.groupby(['Scenario']).mean()
    # Calculate range as ratio of mean
    precision = ((maxbaseCaps - minbaseCaps)/meanbaseCaps/2)
    
    # Find lowest LCOE of each scenario
    minCosts = baseCaps.groupby(['Scenario'])['LCOE ($/MWh)'].min().reset_index()
    # Get base configuration for each scenario as lowest LCOE config
    baseCaps = pd.merge(baseCaps, minCosts, on = ['Scenario', 'LCOE ($/MWh)'], how='inner')
    # Remove duplicates 
    baseCaps = baseCaps[~baseCaps.duplicated()]
    # Assert unique base config for each scenario
    assert baseCaps['Scenario'].value_counts().max() == 1
    
    # calculate range of each field from the baseline(lowest cost) and precision 
    x, y = baseCaps.groupby('Scenario'), baseCaps.groupby('Scenario')
    x = baseCaps.groupby('Scenario').max().multiply(1+precision).reset_index()
    y = baseCaps.groupby('Scenario').max().multiply(1-precision).reset_index()
    # concatenate
    baseCaps = pd.concat([baseCaps, x, y], ignore_index = True)
    del x, y
    
    # Add base scenario ranges as marked new columns - these will shortly be transformed
    # to (percentage/ ) differences
    data = pd.merge(data, baseCaps, on = 'Scenario', how = 'inner', suffixes = ('', '_relative'))
    
    # More convenient lists of relevant columns
    gens_rel = [col + '_relative' for col in gens]
    costs_rel = [col + '_relative' for col in costs]
    
    # Calculate (percentage/ ) differences of generation and cost parameters
    data[costs_rel] = -data[costs_rel].subtract(data[costs].values, axis=0)
    data[gens_rel] = 100-100*data[gens_rel].divide(data[gens].values, axis=0)
    
    # def lambdaReadDeficit(scenario, zone, n_year, event):
    #     try: 
    #         return pd.read_csv(fr'S{scenario}-{zone}-{n_year}-{event}.csv')['eventDeficit'].sum()
    #     except (FileNotFoundError, OSError): 
    #         if pd.isna(zone): 
    #             return 0
    #         else: 
    #             return np.nan
    # data['Energy Deficit (TWh p.a.)'] = data[['Scenario', 'Zone', 'n_year', 'event']].apply(
    #     lambda row: lambdaReadDeficit(*row)*0.5/10/1e6, axis = 1)
    # data['deficit%'] = data['Energy Deficit (TWh p.a.)'] / data['Energy Demand (TWh p.a.)']
    
    # Filter for relevant rows
    data = data.loc[data['event'].isin(events), :]
    data = data.loc[data['Scenario'] == scenario, :]
    
    # Single column to uniquely reference data after transformation
    data['GridZone'] = data['Grid'] + data['Zone'] + data['event']
    
    costs_pres = ['Electricity', 'Generation', 'Storage', 'Transmission', 'Spillage & Loss']
    gens_pres = ['Solar (GW)', 'Wind (GW)', 'Storage (GW)', 'Storage (GWh)']
    
    # reshape data
    data = pd.melt(data, 
                   id_vars = ['GridZone', 'Scenario', 'Zone Name', 'Zone'],
                   value_vars = costs_rel + gens_rel,
                   )
                    
    # Rename strings for presentation
    data['variable'] = data['variable'].str.replace('_relative', '')
    data['variable'] = data['variable'].apply(lambda x: {
        'LCOE ($/MWh)':'Electricity',
        'LCOG ($/MWh)':'Generation',
        'LCOB (storage)':'Storage',
        'LCOB (transmission)':'Transmission',
        'LCOB (spillage & loss)':'Spillage & Loss', 
        'Pumped Hydro (GW)':'Storage (GW)', 
        'Pumped Hydro (GWh)': 'Storage (GWh)',
        }.get(x,x))
    
    data['Zone Name'] = data['Zone Name'].apply(lambda x: x[0])
    data['Zone Name'] = data['Zone Name'].apply(lambda x: {
        'Fitzroy':'Fitzroy (QLD)',
        'Leigh Creek':'Leigh Creek (SA)',
        }.get(x,x))
    
    
    #%%
# =============================================================================
#     Set up plot area
# =============================================================================
    if separate_plots is False:
        fig, axs = plt.subplots(2, figsize=(11,6), dpi=dpi)
        plt.subplots_adjust(hspace = 0.4)
        ax = axs[0]
        ax1 = axs[1]
        
        ax.set_title('a) Energy resources of resilient grids relative to base case')
        ax1.set_title('b) Levelised costs of resilient grids relative to base case')
    else:
        fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
        fig1, ax1 = plt.subplots(figsize=(8,6), dpi=dpi)
        
        ax.set_title('Energy resources of resilient grids relative to base case')    
        ax1.set_title('Levelised costs of resilient grids relative to base case')
    
#%%    
# =============================================================================
#     Figure 6
# =============================================================================
    sns.barplot(
        data = data.loc[data['variable'].isin(gens_pres), :]
        , x = 'Zone Name'
        , y = 'value'
        , hue = 'variable'
        , hue_order = gens_pres
        , order = ['Fitzroy (QLD)', 'Southern NSW Tablelands', 'Leigh Creek (SA)']
        , ax = ax
        )
    ax.axhline(0.0, linewidth = 0.5, color='black')
    
    ax.set_yticks(np.arange(-20, 30, 5))
    ax.grid(which='major', axis='y')
    
    ax = adjust_legend(ax, 1.285, 0.5)
    
    ax.set_xlabel('Zone affected by HILP wind events')
    ax.set_ylabel('Relative Capacity (% GW|GWh)')
        
#%%   
# =============================================================================
#     Figure 7 
# =============================================================================
    sns.barplot(
        data = data.loc[data['variable'].isin(costs_pres), :]
        , x = 'Zone Name'
        , y = 'value'
        , hue = 'variable'
        , hue_order = costs_pres
        , palette = 'Set2'
        , order = ['Fitzroy (QLD)', 'Southern NSW Tablelands', 'Leigh Creek (SA)']
        , ax = ax1
        )
    ax1.axhline(0.0, linewidth = 0.5, color='black')

    ax1.set_yticks(np.arange(-1.5, 2.0, 0.5))
    ax1.grid(which='major', axis='y')

    ax1 = adjust_legend(ax1, 1.285, 0.5)

    ax1.set_xlabel('Zone affected by HILP wind events')
    ax1.set_ylabel('Relative Cost ($/MWh)')

    plt.show()
    
#%%    
    # os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
    os.chdir('graphs')
    
    
    
