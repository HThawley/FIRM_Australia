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


for event in ('e',):#, 's'):
    sfiles = [f for f in os.listdir() if sfileMatch(f)]

    for f in sfiles: 
        if 'All' in f:  eventZone = 'All'
        else: eventZone = '['+re.split('\[|\]', f)[1]+']'
        if eventZone != '[10]': continue
        
        size = (8,3)
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
                     , x = 'eventDeficit'
                      , binwidth = 2500
                     , binrange = [0,data['eventDeficit'].max()]
                     )
        
        ax.set_ylim([0, (ax.get_ylim()[1]//2+1)*2])
        ax.set_xlabel('Half hourly energy deficit (MW)')
        ax.set_ylabel('Occurences per decade')
        ax.set_title(f'Frequency of fragile instances')#' ({scenario}-{eventZone}-{event})')
        plt.show()
        
        

    

os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
os.chdir('graphs')

