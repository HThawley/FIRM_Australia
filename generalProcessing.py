# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:00:48 2023

@author: hmtha
"""

import pandas as pd
import os
import re
import numpy as np
from ast import literal_eval

#%% 

def coalesce(args):
    for x in args: 
        if x is not None: return x
        
def reSearchWrapper(regex, srchstr):
    try: return re.search(regex, srchstr).group()
    except AttributeError: return None
    
def readPrintedArray(txt):      
    if txt == 'None': return None
    txt = re.sub(r"(?<!\[)\s+(?!\])", r",", txt)
    return np.array(literal_eval(txt), dtype =int)
        
    
#%%

def consolidateGGTA(output = False, cap = None):
    assert isinstance(output, bool)
    ggta = pd.DataFrame([])
    active_dir = os.getcwd()
    
    for directory in ('CostOptimisationResults', 'Results'):
        os.chdir(directory)
        
        files = [file for file in os.listdir() if ('GGTA' in file) and ('consolidated' not in file)]
        
        for file in files:   
            df = pd.concat([
                    pd.DataFrame([[int(file[4:6]), 
                                  reSearchWrapper('(\[.*\]|all|None)', file),
                                  reSearchWrapper('(?<=-)\d+(?=-\w.csv)', file), 
                                  reSearchWrapper('(?<=-)\w(?=.csv)', file)]]),
                    pd.read_csv(file, header=None)
                    ], axis = 1)
            ggta = pd.concat([ggta, df])
            
        os.chdir(active_dir)
        
    ggta.columns = ['Scenario', 'Zone', 'n_year', 'event', 'Energy Demand (TWh p.a.)', 
        'HVDC Loss (TWh p.a.)', 'Solar (GW)', 'Solar (TWh p.a.)', 'Wind (GW)', 
        'Wind (TWh p.a.)', 'Hydro & Bio (GW)', 'Hydro & Bio (TWh p.a.)', 
        'Pumped Hydro (GW)', 'Pumped Hydro (GWh)', 'HVDC FNQ-QLD (GW)', 
        'HVDC NSW-QLD (GW)', 'HVDC NSW-SA (GW)', 'HVDC NSW-VIC (GW)', 
        'HVDC NT-SA (GW)', 'HVDC SA-WA (GW)', 'HVDC TAS-VIC (GW)', 'LCOE ($/MWh)',
        'LCOG ($/MWh)', 'LCOB (storage)', 'LCOB (transmission)', 'LCOB (spillage & loss)']
        
    ggta['n_year'] = ggta['n_year'].fillna(-1).astype(int)
    if cap is not None: 
        ggta = addZoneCapacityGGTA(ggta, cap)
    if output: ggta.to_csv('Results/GGTA-consolidated.csv', index = False)

    return ggta.reset_index(drop=True)

def addZoneCapacityGGTA(ggta, cap):
    ggta['zone capacity'] = 0
    
    for scenario in ggta['Scenario'].unique():
        caps = cap[str(scenario)]
        caps = caps.loc[(caps['Zone'] == 'None'), :].to_numpy().reshape((-1,1))[3:]
        
        scMask = ggta['Scenario']==scenario
        zoMask = ggta['Zone'].str.contains('[', regex=False)
        # zoMask=True
        pidx, widx, sidx, headers, szindx = zoneTypeIndx(scenario)

        ggta.loc[scMask & zoMask, 'zone capacity'] = ggta.loc[scMask & zoMask, 'Zone'].apply(readPrintedArray)
        ggta.loc[scMask & zoMask, 'zone capacity'] = ggta.loc[scMask & zoMask, 'zone capacity'].apply(lambda x: zoneTypeIndx(scenario, x)[-1])
        ggta.loc[scMask & zoMask, 'zone capacity'] = ggta.loc[scMask & zoMask, 'zone capacity'].apply(lambda x: caps[x+pidx].sum())
        
        
    return ggta        

#%%

def consolidateCapacities(output=False):
    assert isinstance(output, bool)
    active_dir = os.getcwd()
    caps={}
    
    for directory in ('CostOptimisationResults', 'Results'):
        os.chdir(directory)
    
        files = [file for file in os.listdir() if ('Optimisation_resultx' in file) and ('consolidated' not in file)]
    
        for scenario in pd.Series(files).str.slice(20, 22).unique():
            try: caps[scenario] 
            except KeyError: caps[scenario] = pd.DataFrame([])
            
            for file in files:
                if file[20:22] != scenario: continue
                df = pd.concat([
                    pd.DataFrame([[int(file[20:22]), 
                                   reSearchWrapper('(\[.*\]|all|None)', file),
                                   reSearchWrapper('(?<=-)\d+(?=.csv)', file), 
                                   reSearchWrapper('(?<=-)\w(?=.csv)', file)]]),
                    pd.read_csv(file, header=None)
                    ], axis = 1)
                
                caps[scenario] = pd.concat([caps[scenario], df])
        os.chdir(active_dir)

    for key, df in caps.items():
        pidx, widx, sidx, headers, szindx = zoneTypeIndx(int(key), wdir=active_dir)
        
        df.columns = (['Scenario', 'Zone', 'n_year', 'event'] + headers)
        df['n_year'] = df['n_year'].fillna(-1).astype(int)
        caps[key] = caps[key].reset_index(drop=True)
        
        
    if output: 
        for key, df in caps.items():
            if df.shape[0] > 1:
                df.to_csv('Results/Optimisation_resultx{}-consolidated.csv'.format(key), index = False)
    
    return caps

def zoneTypeIndx(scenario, eventZone=None, wdir=None):
    if wdir is not None: 
        active_dir = os.getcwd()
        os.chdir(wdir)
    Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
    PVl   = np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
    Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)
    names = np.char.replace(np.genfromtxt('Data\ZoneDict.csv', delimiter=',', dtype=str), 'ï»¿', '')
    if str(eventZone) == 'all': eventZone = ...

    if scenario<=17:
        node = Nodel[scenario % 10]
        names = names [np.where(np.append(PVl, Windl)==node)[0]] 
        pzones = len(np.where(PVl==node)[0])
        wzones = len(np.where(Windl==node)[0])
        coverage = np.array([node])
        
        zones = np.zeros(Windl.shape)
        zones[eventZone] = 1 
        zones = zones[np.where(Windl==node)[0]]
        eventZoneIndx = np.where(zones==1)[0]
        
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
        zones[eventZone] = 1 
        zones = zones[np.where(np.in1d(Windl, coverage)==True)[0]]
        eventZoneIndx = np.where(zones==1)[0]
        
        names = names[np.where(np.in1d(np.append(PVl, Windl), coverage)==True)[0]]
        pzones = len(np.where(np.in1d(PVl, coverage)==True)[0])
        wzones = len(np.where(np.in1d(Windl, coverage)==True)[0])
       
    nodes = len(coverage)
    pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)
    
    headers = (['pv-'   + name + ' (GW)' for name in names[:pidx]] + 
               ['wind-' + name + ' (GW)' for name in names[pidx:widx]] + 
               ['storage-' + name + ' (GW)' for name in coverage] + 
               ['storage (GWh)'])
    
    if eventZone is None: eventZoneIndx=None
    
    if wdir is not None: 
        os.chdir(active_dir)
    return pidx, widx, sidx, headers, eventZoneIndx


#%%
if __name__ == '__main__':
    os.chdir(r'C:\Users\hmtha\Desktop\FIRM_Australia')
    output = True
    cap = consolidateCapacities(output)
    ggta = consolidateGGTA(output, cap)

