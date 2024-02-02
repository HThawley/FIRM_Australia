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

from graphutils import directory_up, reSearchWrapper

#%%
# =============================================================================
# auxiliary functions
# =============================================================================
def sfileMatch(file):
    try: 
        assert file[:4] == f'S{scenario}-'
        assert reSearchWrapper('(?<=-)\d+(?=-)', file) == str(n_year)
        assert reSearchWrapper('(?<=-)(\w)(?=(-\d+)?.csv)', file) in events
        ez = 'All' if 'All' in file else '['+re.split('\[|\]', file)[1]+']'
        assert ez in eventZones
        return True
    except (IndexError, AssertionError): 
        return False
    
def removeYear(date): return dt(2000, date.month, date.day, date.hour, date.minute)
def lambdaDt(*args): return dt(*args)

#%%
# =============================================================================
# #parameters
# =============================================================================
scenario = 21
n_year = 25
events = ('e',)
dpi = 100
eventZones = ('[10]',)

# manage directory 
directory_up()
os.chdir('Results')


sfiles = [f for f in os.listdir() if sfileMatch(f)]

for f in sfiles: 
   
    data = pd.read_csv(f) 

    data['eventEnergyDeficit'] = data['eventDeficit'] * 0.5 # MW to MWh
    data['eventDeficit%'] = 100*data['eventDeficit'] / data['Operational demand (original)']
    
    fig, ax = plt.subplots(figsize=(8,3), dpi=dpi)
    
    sns.histplot(data = data.loc[data['eventDeficit'] != 0, :],
                 x = 'eventDeficit',
                 binwidth = 2500,
                 binrange = [0,data['eventDeficit'].max()],
                 )
    
    ax.set_ylim([0, (ax.get_ylim()[1]//2+1)*2])
    ax.set_xlabel('Half hourly energy deficit (MW)')
    ax.set_ylabel('Occurences per decade')
    ax.set_title(f'Frequency of fragile instances ({f[1:-4]})')
    plt.show()

# Reset directory
directory_up()
os.chdir('graphs')

