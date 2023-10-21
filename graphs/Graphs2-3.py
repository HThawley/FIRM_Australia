# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:30:40 2023

@author: hmtha
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import cm
from matplotlib import dates as pltd
import os
import re
from datetime import datetime as dt

def sfileMatch(file):
    try: 
        assert file[:4] == f'S{scenario}-'
        assert file[-9:] == f'-{n_year}-{event}.csv'
        return True
    except (IndexError, AssertionError): return False
    return False
    
def removeYear(date): return dt(2000, date.month, date.day, date.hour, date.minute)
def lambdaDt(*args): return dt(*args)

scenario = 21
n_year = 25
# event = 'd'


os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
os.chdir('Results')

ggtaSummary = pd.read_csv("GGTA-consolidated.csv")
# ggtaSummary = pd.read_csv(r"C:\Users\hmtha\Desktop\GGTA-consolidated.csv")


for event in ('d'):#, 's'):
    sfiles = [f for f in os.listdir() if sfileMatch(f)]

    for f in sfiles: 
        if 'All' in f:  eventZone = 'All'
        else: eventZone = '['+re.split('\[|\]', f)[1]+']'
        
        size = (9,4)
        data = pd.read_csv(f) 
        # data['Date & time'] = pd.to_datetime(data['Date & time'], format='%a -%d %b %Y %H:%M')
        # data['Day'] = data['Date & time'].dt.day
        # data['Month'] = data['Date & time'].dt.month
        # data['Year'] = data['Date & time'].dt.year
        # data = data.drop(columns = ['Date & time'])
        # data = data.groupby(['Year', 'Month', 'Day']).sum().reset_index()
        # data['eventDeficit%'] = 100*data['eventDeficit'] / data['Operational demand (original)']
        
        data['eventEnergyDeficit'] = data['eventDeficit'] * 0.5
        data['eventDeficit%'] = 100*data['eventDeficit'] / data['Operational demand (original)']
        
        fig, ax = plt.subplots(figsize=size)
        sns.histplot(data = data.loc[data['eventDeficit'] != 0, :]
                     , x = 'eventDeficit%'
                     , binwidth = 5
                     , binrange = [0,data['eventDeficit%'].max()]
                     )
        
        ax.set_ylim([0, (ax.get_ylim()[1]//2+1)*2])
        ax.set_xlabel('Half Hourly Energy deficit (%)')
        ax.set_ylabel('Occurences per decade')
        ax.set_title(f'Frequency of fragile instances')#' ({scenario}-{eventZone}-{event})')
        plt.show()
        
        

        data['Date & time'] = pd.to_datetime(data['Date & time'], format='%a -%d %b %Y %H:%M')
    
        data['Day'] = data['Date & time'].dt.day
        data['Month'] = data['Date & time'].dt.month
        data['Year'] = data['Date & time'].dt.year
        data = data.drop(columns = ['Date & time'])
        
        # data = (data.groupby(['Year', 'Month', 'Day']).sum()).reset_index()
        
        # data['energyDeficit%'] = 100*data['eventDeficit'] / data['Operational demand (original)']
        
        # data = (data.groupby(['Month', 'Day']).sum()/10).reset_index()
        datamean = (data.groupby(['Month', 'Day']).mean()).reset_index()
        data = (data.groupby(['Month', 'Day']).max()).reset_index()
        
        data['Date'] = data[['Month','Day']].apply(lambda x: dt(2000, *x), axis = 1)
        datamean['Date'] = datamean[['Month','Day']].apply(lambda x: dt(2000, *x), axis = 1)
        # sns.lineplot(data = data
        #                 , x = 'Date' 
        #                 , y = 'eventDeficit%' 
        #                 )
        fig, ax = plt.subplots(figsize=size)
        ax.stem(
            data['Date']
            , data['eventDeficit%']
            , basefmt= ' '
            , markerfmt = ' '
            )
        ax.axhline(0.0, linewidth=1, color = sns.color_palette()[0])
    
        # ax.set_xticks([dt(2000, m, 1) for m in range(1,13,2)] + [dt(2000, 12, 31)])
        ax.xaxis.set_major_formatter(pltd.DateFormatter('%d-%b'))
        ax.set_xticks([dt(2000,i,1) for i in range(1, 13)] + [dt(2000,12,31)])
        ax.set_ylim([0, (ax.get_ylim()[1]//2+1)*2])
        ax.set_xlabel('Date')
        ax.set_ylabel('Power deficit (%)')
        ax.set_title(f'Maximum half-hourly power deficit by date of event')#' ({scenario}-{eventZone}-{event})')
        
        # sns.scatterplot(data = data, 
        #                 x = 'Date & time', 
        #                 y = 'eventDeficit')
        # ax.xaxis.set_major_formatter(pltd.DateFormatter('%Y'))
    

os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
os.chdir('graphs')
#%%
def cfgraph(source='pv'):
    
    import matplotlib.pyplot as plt 
    from matplotlib import dates as pltd
    # import seaborn as sns
    from datetime import datetime as dt
    

    
    df = pd.read_csv(f'Data/{source}.csv')
    df = df.groupby(['Month', 'Day']).mean().reset_index()
    
    df2 = pd.DataFrame(df[['Month', 'Day']].apply(lambda x: dt(2000,*x), axis = 1), columns=['dt'])
    df2['cf'] = 100*df.iloc[:,4:].mean(axis = 1)
    
    fig, ax = plt.subplots(figsize = (9,4))
    

    ax.stem(
        df2['dt']
        , df2['cf']
        , basefmt= ' '
        , markerfmt = ' '
        , label = 'capacity factor'
        )
    
    ax.xaxis.set_major_formatter(pltd.DateFormatter('%d-%b'))
    ax.set_xticks([dt(2000,i,1) for i in range(1, 13)] + [dt(2000,12,31)])
    ax1 = ax.twinx()
    ax1.stem(
        data['Date']
        , data['eventDeficit%']
        , basefmt= ' '
        , markerfmt = ' '
        , linefmt = 'r'
        , label = 'Power Deficit'
        )
    ax.axhline(0.0, linewidth=1, color = sns.color_palette()[0])
    

    ax1.plot(datamean['Date'], 100*datamean['PHES-Storage']/ datamean['PHES-Storage'].max()
             , color = 'green', label = 'State of storage')
    ax1.plot(datamean['Date'], 
             100*datamean['Operational demand (original)']/ datamean['Operational demand (original)'].max(),
             color='black', label = 'Demand')
    
    if source == 'pv': source, no = 'photovoltaic', 'a)' 
    if source == 'wind': no = 'b)'
    ax.set_title(f'{no} Fragile instances and daily mean {source} capacity factor, state of storage, power demand.')
    ax.set_ylabel('Capacity factor (%)')
    ax.set_xlabel('Date')
    ax.set_ylim(-5,105)
    ax1.set_ylabel('Power and energy (% of maximum)')
    
    plt.figlegend(loc='lower left')
    plt.show()

os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
cfgraph('pv')
cfgraph('wind')
os.chdir('graphs')
