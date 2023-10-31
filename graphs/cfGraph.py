# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:40:49 2023

@author: hmtha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import dates as pltd
import seaborn as sns
from datetime import datetime as dt
import os
from ast import literal_eval
import re


def readPrintedArray(txt):      
    if txt == 'None': return None
    txt = re.sub(r"(?<!\[)\s+(?!\])", r",", txt)
    return np.array(literal_eval(txt), dtype =int)

def cfWeightings(scenario, eventZone, capacities):
    Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
    PVl   = np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
    Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

    if isinstance(eventZone, str):        
        if str(eventZone) == 'all': eventZone = ...
        else: eventZone = readPrintedArray(eventZone) 

    if scenario<=17:
        node = Nodel[scenario % 10]
        pzones = len(np.where(PVl==node)[0])
        wzones = len(np.where(Windl==node)[0])
        coverage = np.array([node])
        
        pWeighting = np.zeros(PVl.shape)
        wWeighting = np.zeros(Windl.shape)
        
        pCaps = capacities[:pzones]
        wCaps = capacities[pzones:pzones+wzones]
        
        pWeighting[np.where(PVl==node)] = pCaps/pCaps.sum()
        wWeighting[np.where(Windl==node)] = wCaps/wCaps.sum()
        

    if scenario>=21:
        coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1]
        
        pzones = len(np.where(np.in1d(PVl, coverage)==True)[0])
        wzones = len(np.where(np.in1d(Windl, coverage)==True)[0])
       
        pWeighting = np.zeros(PVl.shape)
        wWeighting = np.zeros(Windl.shape)
        
        pCaps = capacities[:pzones]
        wCaps = capacities[pzones:pzones+wzones]
        
        pWeighting[np.where(np.in1d(PVl, coverage)==True)] = pCaps/pCaps.sum()
        wWeighting[np.where(np.in1d(Windl, coverage)==True)] = wCaps/wCaps.sum()
    return pWeighting, wWeighting 

#%%
def cfgraphWrapper(source='both', weighting = None): 
        
    if source != 'both':
        fig, ax = plt.subplots(figsize = (9,4))
        cfgraph(ax, source, weighting)
        plt.figlegend(loc='lower left')
        plt.show()
        return
    
    fig, axs = plt.subplots(2, figsize = (10,7))
    plt.subplots_adjust(hspace = 0.42)
    
    
    cfgraph(axs[0], 'Solar', weighting[0])
    cfgraph(axs[1], 'Wind', weighting[1])
    
    plt.show()
    return
    

def cfgraph(ax, source, weighting):
    
    if source == 'Solar':
        df = pd.read_csv('Data/pv.csv')
    if source == 'Wind':
        df = pd.read_csv('Data/wind.csv')

    if year is not None: df = df[df['Year'] == year]
    df = df.groupby(['Month', 'Day']).mean().reset_index()
    
    cdf = pd.DataFrame(df[['Month', 'Day']].apply(lambda x: dt(2000,*x), axis = 1), columns=['dt'])
    if weighting is not None: cdf['cf'] = 100*df.iloc[:,4:].multiply(weighting).sum(axis = 1)
    else: cdf['cf'] = 100*df.iloc[:,4:].sum(axis = 1)
    
    if source == 'Solar': colo = sns.color_palette()[0]
    if source == 'Wind': colo = sns.color_palette()[1]
    
    l = ax.plot(
        cdf['dt']
        , cdf['cf']
        # , basefmt= ' '
        # , markerfmt = ' '
        # , label = 'Capacity factor'
        )
    l[0].set_color(colo)

    ax.fill_between(
        cdf['dt']
        , cdf['cf']
        , 0
        , label = f'{source} capacity factor'
        , color = colo
        )
    
    ax.xaxis.set_major_formatter(pltd.DateFormatter('%d-%b'))
    ax.set_xticks([dt(2000,i,1) for i in range(1, 13)] + [dt(2000,12,31)])
    # ax1 = ax.twinx()
    ax.stem(
        sdataMax['Date']
        , sdataMax['eventDeficit%']
        , basefmt= ' '
        , markerfmt = ' '
        , linefmt = 'black'
        , label = 'Maximum Power Deficit'
        )
    ax.axhline(0.0, linewidth=1, color = sns.color_palette()[0])
    

    # ax.plot(sdataMean['Date'], 100*sdataMean['PHES-Storage']/ sdataMean['PHES-Storage'].max()
    #          , color = 'green', label = 'State of storage')
    ax.plot(sdataMean['Date'], 
             100*sdataMean['Operational demand (original)']/ sdataMean['Operational demand (original)'].max(),
             color='cyan', label = 'Demand')
    
    if source == 'Solar': no =  'a)' 
    if source == 'Wind': no = 'b)'
    ax.set_title(f'{no} 10-year daily maximum power deficit, 10-year daily mean power demand,\nand 10-year daily weighted mean capacity factor of {source.lower()} generation.')
    ax.set_ylabel('Maximum deficit, demand,\n& capacity factor(%)')
    ax.set_xlabel('Date')
    ax.set_ylim(-5,105)
    # ax.set_ylabel('Power and energy (% of maximum)')
    
    
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.95, pos.height])
    
    lns, labs = ax.get_legend_handles_labels()
    # lns2, labs2 = ax1.get_legend_handles_labels()
    
    ax.legend(
        lns #+ lns2
        , labs #+ labs2
        , loc = 'center right'
        , bbox_to_anchor = (1.33, 0.5)
        )
    


#%%

if __name__ == '__main__': 
    
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
    
    scenario = 21
    n_year = 25
    eventZone = np.array([10])
    event = 'e'
    year = None
    
    sdata = pd.read_csv(f'Results/S{scenario}-{eventZone}-{n_year}-{event}.csv')
    capacities = np.genfromtxt(f'Results/Optimisation_resultx{scenario}-{eventZone}-{n_year}-{event}.csv',dtype=float, delimiter=',')
    
    pw, ww = cfWeightings(scenario, eventZone, capacities)
    
    sdata['eventDeficit%'] = 100*sdata['eventDeficit'] / sdata['Operational demand (original)']
    sdata['Date & time'] = pd.to_datetime(sdata['Date & time'], format='%a -%d %b %Y %H:%M')
        
    sdata['Day'] = sdata['Date & time'].dt.day
    sdata['Month'] = sdata['Date & time'].dt.month
    sdata['Year'] = sdata['Date & time'].dt.year
    sdata = sdata.drop(columns = ['Date & time'])
    
    if year is not None: sdata = sdata[sdata['Year'] == year]
    
    sdataMax = sdata.groupby(['Month', 'Day']).max().reset_index()
    sdataMean = sdata.groupby(['Month', 'Day']).mean().reset_index()
    
    sdataMax['Date'] = sdataMax[['Month','Day']].apply(lambda x: dt(2000, *x), axis = 1)
    sdataMean['Date'] = sdataMean[['Month','Day']].apply(lambda x: dt(2000, *x), axis = 1)

    
    cfgraphWrapper('both', (pw, ww))

    os.chdir('graphs')