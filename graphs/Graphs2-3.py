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


for event in ('d', 's'):
    sfiles = [f for f in os.listdir() if sfileMatch(f)]

    for f in sfiles: 
        if 'All' in f:  eventZone = 'All'
        else: eventZone = '['+re.split('\[|\]', f)[1]+']'
        
        data = pd.read_csv(f) 
        data['eventEnergyDeficit'] = data['eventDeficit'] * 0.5
        data['eventDeficit%'] = 100*data['eventDeficit'] / data['Operational demand (original)']
        
        fig, ax = plt.subplots() 
        sns.histplot(data = data.loc[data['eventDeficit'] != 0, :]
                     , x = 'eventDeficit%'
                     , binwidth = 5
                     , binrange = [0,data['eventDeficit%'].max()]
                     )
        ax.set_xlabel('Half Hourly Energy deficit (%)')
        ax.set_ylabel('Occurences per decade')
        ax.set_title(f'Number of fragile times ({scenario}-{eventZone}-{event})')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        data['Date & time'] = pd.to_datetime(data['Date & time'], format='%a -%d %b %Y %H:%M')
    
        data['Day'] = data['Date & time'].dt.day
        data['Month'] = data['Date & time'].dt.month
        data = data.drop(columns = ['Date & time'])
        # data = (data.groupby(['Month', 'Day']).sum()/10).reset_index()
        data = (data.groupby(['Month', 'Day']).max()).reset_index()
        
        data['Date'] = data[['Month','Day']].apply(lambda x: dt(2000, *x), axis = 1)
    
        # sns.lineplot(data = data
        #                 , x = 'Date' 
        #                 , y = 'eventDeficit%' 
        #                 )
        
        ax.stem(
            data['Date']
            , data['eventDeficit%']
            , basefmt= ' '
            , markerfmt = ' '
            )
        ax.axhline(0.0, linewidth=1, color = sns.color_palette()[0])
    
        # ax.set_xticks([dt(2000, m, 1) for m in range(1,13,2)] + [dt(2000, 12, 31)])
        ax.xaxis.set_major_formatter(pltd.DateFormatter('%d-%b'))
    
        ax.set_xlabel('Date (daily)')
        ax.set_ylabel('Mean energy deficit (%)')
        ax.set_title(f'Expected deficit by date of event ({scenario}-{eventZone}-{event})')
        
        # sns.scatterplot(data = data, 
        #                 x = 'Date & time', 
        #                 y = 'eventDeficit')
        # ax.xaxis.set_major_formatter(pltd.DateFormatter('%Y'))
    

os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
os.chdir('graphs')

