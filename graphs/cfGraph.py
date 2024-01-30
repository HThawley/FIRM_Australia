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
#%%

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
        
    if source in ('Solar', 'Wind'):
        fig, ax = plt.subplots(figsize = (9,4))
        cfgraph(ax, source, weighting)
        plt.figlegend(loc='lower left')
        plt.show()
        return
    
    if source == 'both':
        fig, axs = plt.subplots(2, figsize = (8,6))
        plt.subplots_adjust(hspace = 0.7)
        
        cfgraph(axs[0], 'Solar', weighting[0])
        cfgraph(axs[1], 'Wind', weighting[1])
        
        plt.show()
        return
    
    if source == 'combined':
        fig, ax = plt.subplots(figsize=(8,6), dpi=2500)
        cfgraphComb(ax, weighting)
        
        
def cfgraphComb(ax, weighting):
    
    s = pd.read_csv('Data/pv.csv')
    w = pd.read_csv('Data/wind.csv')
    
    if year is not None: 
        s = s[s['Year'] == year]
        w = w[w['Year'] == year]
    s = s.groupby(['Month', 'Day']).mean().reset_index()
    w = w.groupby(['Month', 'Day']).mean().reset_index()

    cs = pd.DataFrame(s[['Month', 'Day']].apply(lambda x: dt(2000,*x), axis = 1), columns=['dt'])
    cw = pd.DataFrame(w[['Month', 'Day']].apply(lambda x: dt(2000,*x), axis = 1), columns=['dt'])
    if weighting is not None: 
        cs['cf'] = 100*s.iloc[:,4:].multiply(weighting[0]).sum(axis = 1)
        cw['cf'] = 100*w.iloc[:,4:].multiply(weighting[1]).sum(axis = 1)
    else: 
        cs['cf'] = 100*s.iloc[:,4:].sum(axis = 1)
        cw['cf'] = 100*w.iloc[:,4:].sum(axis = 1)
        
    ax.plot(
        cs['dt']
        , cs['cf']
        # , 0
        , label = 'Solar capacity\nfactor (%)'
        , color = sns.color_palette()[0]
        , lw = 2
        )
    ax.plot(
        cs['dt']
        , cw['cf'] #+ cs['cf'] 
        # , cs['cf']
        , label = 'Wind capacity\nfactor (%)'
        , color = sns.color_palette()[1]
        , lw = 2
        )
    
    ax.xaxis.set_major_formatter(pltd.DateFormatter('%d-%b'))
    ax.set_xticks([dt(2000,i,1) for i in range(1, 13)] + [dt(2000,12,31)])
    # ax1 = ax.twinx()
    ls = ax.stem(
        sdataMax['Date']
        , sdataMax['eventDeficit%']
        , basefmt= ' '
        , markerfmt = ' '
        , linefmt = 'black'
        , label = 'Maximum Power\nDeficit (%)'
        )
    ls[1].set_linewidth(2)
    
    ax.axhline(0.0, linewidth=1, color = 'black')
    ax.set_title('10-year daily maximum power deficit due to HILP events in Fitzroy QLD\nand 10-year daily weighted mean capacity factor of renewable generation.')
    ax.set_ylabel('Maximum deficit (%)\n& capacity factor (%)')
    ax.set_xlabel('Date')
    ax.set_ylim([0,None])

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.95, pos.height])
    
    lns, labs = ax.get_legend_handles_labels()
    # lns2, labs2 = ax1.get_legend_handles_labels()
    
    ax.legend(
        lns #+ lns2
        , labs #+ labs2
        , loc = 'center right'
        , bbox_to_anchor = (1.3, 0.5)
        )

def cfgraph(ax, source, weighting):
    
    if source == 'Solar': df = pd.read_csv('Data/pv.csv')
    if source == 'Wind': df = pd.read_csv('Data/wind.csv')

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
        )
    l[0].set_color(colo)

    ax.fill_between(
        cdf['dt']
        , cdf['cf']
        , 0
        , label = f'{source} capacity\nfactor (%)'
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
        , label = 'Maximum Power\nDeficit (%)'
        )
    ax.axhline(0.0, linewidth=1, color = colo)
    

    # ax.plot(sdataMean['Date'], 100*sdataMean['PHES-Storage']/ sdataMean['PHES-Storage'].max()
    #          , color = 'green', label = 'State of storage')
    # ax.plot(sdataMean['Date'], 
    #          100*sdataMean['Operational demand (original)']/ sdataMean['Operational demand (original)'].max(),
    #          color='cyan', label = 'Demand (%)')
    
    if source == 'Solar': no =  'a)' 
    if source == 'Wind': no = 'b)'
    ax.set_title(f'{no} 10-year daily maximum power deficit\nand 10-year daily weighted mean capacity factor of {source.lower()} generation.')
    ax.set_ylabel('Maximum deficit (%)\n& capacity factor (%)')
    ax.set_xlabel('Date')
    ax.set_ylim(-5,105)
    # ax.set_ylabel('Power and energy (% of maximum)')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.95, pos.height])
    
    lns, labs = ax.get_legend_handles_labels()
    # lns2, labs2 = ax1.get_legend_handles_labels()
    
    ax.legend(
        lns #+ lns2
        , labs #+ labs2
        , loc = 'center right'
        , bbox_to_anchor = (1.3, 0.5)
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

    
    cfgraphWrapper('combined', (pw, ww))

    os.chdir('graphs')
    
#%%

# sdata = pd.read_csv(f'Results/S{scenario}-{eventZone}-{n_year}-{event}.csv')
# capacities = np.genfromtxt(f'Results/Optimisation_resultx{scenario}-{eventZone}-{n_year}-{event}.csv',dtype=float, delimiter=',')

# sdata['eventDeficit%'] = 100*sdata['eventDeficit'] / sdata['Operational demand (original)']
# sdata['Date & time'] = pd.to_datetime(sdata['Date & time'], format='%a -%d %b %Y %H:%M')
    

# sdata['Minute'] = sdata['Date & time'].dt.minute
# sdata['Hour'] = sdata['Date & time'].dt.hour
# sdata['Day'] = sdata['Date & time'].dt.day
# sdata['Month'] = sdata['Date & time'].dt.month
# sdata['Year'] = sdata['Date & time'].dt.year
# sdata = sdata.drop(columns = ['Date & time'])

# sdata['eventDeficit%'] = sdata['eventDeficit%'].apply(lambda x: np.nan if x == 0 else x)

# sdataMax = sdata.groupby(['Hour', 'Minute']).max().reset_index()
# sdataCount = sdata.groupby(['Hour', 'Minute']).count().reset_index()

# sdataMax['Date'] = sdataMax[['Hour','Minute']].apply(lambda x: dt(2000,1,1, *x), axis = 1)
# sdataCount['Date'] = sdataCount[['Hour','Minute']].apply(lambda x: dt(2000,1,1, *x), axis = 1)

# fig, ax = plt.subplots()
# ax.plot(
#     sdataCount['Date'],
#     sdataCount['eventDeficit%']
#     )

# ax.xaxis.set_major_formatter(pltd.DateFormatter('%H:%M'))
