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



scenario = 21
n_year = 25

def sfileMatch(file):
    try: 
        assert file[:4] == f'S{scenario}-'
        assert file[-7:] == f'-{n_year}.csv'
        return True
    except (IndexError, AssertionError): return False
    return False
    
def removeYear(date): return dt(2000, date.month, date.day, date.hour, date.minute)
def lambdaDt(*args): return dt(*args)


os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
os.chdir('Results')

ggtaSummary = pd.read_csv("GGTA-consolidated.csv")
# ggtaSummary = pd.read_csv(r"C:\Users\hmtha\Desktop\GGTA-consolidated.csv")

sfiles = [f for f in os.listdir() if sfileMatch(f)]

for f in sfiles: 
    if 'All' in f:  stormZone = 'All'
    else: stormZone = '['+re.split('\[|\]', f)[1]+']'
    
    data = pd.read_csv(f) 
    data['StormEnergyDeficit'] = data['StormDeficit'] * 0.5
    data['StormDeficit%'] = 100*data['StormDeficit'] / data['Operational demand (original)']
    
    fig, ax = plt.subplots() 
    sns.histplot(data = data.loc[data['StormDeficit'] != 0, :]
                 ,x = 'StormDeficit%')
    ax.set_xlabel('Half Hourly Energy deficit (%)')
    ax.set_ylabel('Occurences per decade')
    ax.set_title('Frequency of energy deficits')
    plt.show()
    
    fig, ax = plt.subplots()
    data['Date & time'] = pd.to_datetime(data['Date & time'], format='%a -%d %b %Y %H:%M')

    data['Day'] = data['Date & time'].dt.day
    data['Month'] = data['Date & time'].dt.month
    data = data.drop(columns = ['Date & time'])
    data = (data.groupby(['Month', 'Day']).sum()/10).reset_index()
    # data = (data.groupby(['Month', 'Day']).sum()).reset_index()
    
    data['Date'] = data[['Month','Day']].apply(lambda x: dt(2000, *x), axis = 1)
    sns.lineplot(data = data
                    , x = 'Date' 
                    , y = 'StormDeficit%' 
                    )
    # ax.set_xticks([dt(2000, m, 1) for m in range(1,13,2)] + [dt(2000, 12, 31)])
    ax.xaxis.set_major_formatter(pltd.DateFormatter('%d-%b'))

    ax.set_xlabel('Date (daily)')
    ax.set_ylabel('Mean energy deficit (%)')
    ax.set_title('Annual pattern of power deficits')
    
    # sns.scatterplot(data = data, 
    #                 x = 'Date & time', 
    #                 y = 'StormDeficit')
    # ax.xaxis.set_major_formatter(pltd.DateFormatter('%Y'))
    

os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
os.chdir('graphs')

