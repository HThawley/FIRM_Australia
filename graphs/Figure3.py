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

from graphutils import readPrintedArray, manage_nodes, directory_up, adjust_legend

#%%
def capacity_weightings(scenario, capacities):
    """Get capacity factors to generate weightings"""
    node_data = manage_nodes(scenario)
    Nodel, PVl, Windl = node_data[0]
    pzones, wzones = node_data[1]
    coverage = node_data[3]

    if scenario<=17:
        pWeighting = np.zeros(PVl.shape)
        wWeighting = np.zeros(Windl.shape)
        
        pCaps = capacities[:pzones]
        wCaps = capacities[pzones:pzones+wzones]
        
        pWeighting[np.where(PVl==coverage[0])] = pCaps/pCaps.sum()
        wWeighting[np.where(Windl==coverage[0])] = wCaps/wCaps.sum()

    if scenario>=21:       
        pWeighting = np.zeros(PVl.shape)
        wWeighting = np.zeros(Windl.shape)
        
        pCaps = capacities[:pzones]
        wCaps = capacities[pzones:pzones+wzones]
        
        pWeighting[np.where(np.in1d(PVl, coverage)==True)] = pCaps/pCaps.sum()
        wWeighting[np.where(np.in1d(Windl, coverage)==True)] = wCaps/wCaps.sum()
        
    return pWeighting, wWeighting 

#%%
def graphWrapper(): 
    if separate_plots is True:
        fig, ax = plt.subplots(figsize = (9,4), dpi=dpi)
        plot_axis(ax, 'Solar', sw)
        plt.figlegend(loc='lower left')
        
        fig1, ax1 = plt.subplots(figsize = (9,4), dpi=dpi)
        plot_axis(ax1, 'Wind', ww)
        plt.figlegend(loc='lower left')
        return
    
    if separate_plots == 'both':
        fig, axs = plt.subplots(2, figsize=(8,6), dpi=dpi)
        plt.subplots_adjust(hspace = 0.7)
        
        plot_axis(axs[0], 'Solar', sw)
        plot_axis(axs[1], 'Wind', ww)
        
        return
    
    if separate_plots == 'combined':
        fig, ax = plt.subplots(figsize=(8,6), dpi=dpi)
        plot_axis(ax, 'Solar', sw)
        plot_axis(ax, 'Wind', ww)
        return

def plot_axis(ax, source, weighting):

    # Get long term capacity factor data
    if source == 'Solar': 
        df = pd.read_csv('Data/pv.csv')
        colo = sns.color_palette()[0]
        no = 'a)' if separate_plots == 'both' else '' 
    if source == 'Wind': 
        df = pd.read_csv('Data/wind.csv')
        colo = sns.color_palette()[1]
        no = 'b)' if separate_plots == 'both' else '' 

    # Filter for specific years
    if years is not None: 
        df = df[df['Year'].isin(years)]
    # Average capacity factors 
    df = df.groupby(['Month', 'Day']).mean().reset_index()
    
    # create new data frame with date column to be used for graph
    if weighting is not None: 
        df = pd.concat(
            [pd.DataFrame(df[['Month', 'Day']].apply(lambda x: dt(2000,*x), axis = 1)), 
             100*df.iloc[:,4:].multiply(weighting).sum(axis = 1)], 
            axis = 1, 
            ignore_index=True)
    else: 
        df = pd.concat(
            [pd.DataFrame(df[['Month', 'Day']].apply(lambda x: dt(2000,*x), axis = 1)), 
             100*df.iloc[:,4:].sum(axis = 1)], 
            axis = 1, 
            ignore_index=True)
    df=df.rename(columns={0:'dt',1:'cf'})
        

    # Colour in area graph if no combined
    if separate_plots != 'combined':
        # Plot trace    
        l = ax.plot(df['dt'], df['cf'])
        # Fix colour 
        l[0].set_color(colo)
        ax.fill_between(df['dt'], df['cf'], 
            label = f'{source} capacity\nfactor (%)',
            color = colo)
        ax.axhline(0.0, linewidth=1, color = colo)
    else: 
        # Plot trace    
        l = ax.plot(df['dt'], df['cf'],label = f'{source} capacity\nfactor (%)')
        # Fix colour 
        l[0].set_color(colo)
        ax.axhline(0.0, linewidth=1, color = 'black')
    
    # Plot deficits
    ax.stem(
        deficits['Date'],
        deficits['eventDeficit%'],
        basefmt= ' ',
        markerfmt = ' ',
        linefmt = 'black',
        label = 'Mean Power\nDeficit (%)',
        )
    
    # Format Axis for presentation
    ax.xaxis.set_major_formatter(pltd.DateFormatter('%d-%b'))
    ax.set_xticks([dt(2000,i,1) for i in range(1, 13)] + [dt(2000,12,31)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Titles and labels
    if separate_plots != 'combined':
        ax.set_title(f'{no} 10-year daily maximum power deficit\nand 10-year daily weighted mean capacity factor of {source.lower()} generation.')
    else: 
        ax.set_title('10-year daily maximum power deficit\nand 10-year daily weighted mean capacity factor of generation.')
    ax.set_ylabel('Maximum deficit (%)\n& capacity factor (%)')
    ax.set_xlabel('Date')
    ax.set_ylim(-5,105)
    
    if separate_plots == 'both':
        dim = 1.3, 0.5 
    if separate_plots == 'combined':
        dim = 1.33, 0.5 
    if separate_plots is True:
        dim = 1.3, 0.5
    
    adjust_legend(ax, *dim)
    
    
def find_deficits():
    sdata = pd.read_csv(f'Results/S{scenario}-{eventZone}-{n_year}-{event}.csv')

    sdata['eventDeficit%'] = 100*sdata['eventDeficit'] / sdata['Operational demand (original)']

    sdata['Date & time'] = pd.to_datetime(sdata['Date & time'], format='%a -%d %b %Y %H:%M')
    sdata['Day'] = sdata['Date & time'].dt.day
    sdata['Month'] = sdata['Date & time'].dt.month
    sdata['Year'] = sdata['Date & time'].dt.year
    sdata = sdata.drop(columns = ['Date & time'])
    
    if years is not None: 
        sdata = sdata[sdata['Year'].isin(years)]
    
    sdataMax = sdata.groupby(['Month', 'Day']).mean().reset_index()
    
    sdataMax['Date'] = sdataMax[['Month','Day']].apply(lambda x: dt(2000, *x), axis = 1)
    
    return sdataMax[['Date', 'eventDeficit%']]
    

#%%
if __name__ == '__main__': 
    
    scenario = 21
    n_year = 25
    eventZone = np.array([7])
    event = 'e'
    years = None
    # how to separate plots: 'both', 'combined', True
    separate_plots = True
    dpi=100
    
    directory_up()
    capacities = np.genfromtxt(f'Results/Optimisation_resultx{scenario}-{eventZone}-{n_year}-{event}.csv', dtype=float, delimiter=',')
    
    sw, ww = capacity_weightings(scenario, capacities)
    
    deficits = find_deficits()

    graphWrapper()

    os.chdir('graphs')
    
