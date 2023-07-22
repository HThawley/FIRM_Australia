# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:00:48 2023

@author: hmtha
"""

import pandas as pd
import os
import re



#%%

def consolidateGGTA(directory='c', output = False):
    if directory in ('cost', 'c'): directory = 'CostOptimisationResults'
    if directory in ('result', 'r'): directory = 'Results'
        
    active_dir = os.getcwd()
    os.chdir(directory)
    
    ggta = pd.DataFrame([])
    for file in os.listdir():
        if 'GGTA' not in file: continue
        if 'consolidated' in file: continue
        
        df = pd.DataFrame([int(file[4:6])])
    
        df = pd.concat([df, pd.read_csv(file, header=None)], axis = 1)
        
        ggta = pd.concat([ggta, df])
    
    del df, file
    
    ggta.columns = ['Scenario', 'Energy Demand (TWh p.a.)', 'HVDC Loss (TWh p.a.)', 'Solar (GW)', 
        'Solar (TWh p.a.)', 'Wind (GW)', 'Wind (TWh p.a.)', 'Hydro & Bio (GW)', 
        'Hydro & Bio (TWh p.a.)', 'Pumped Hydro (GW)', 'Pumped Hydro (GWh)',
        'HVDC FNQ-QLD (GW)', 'HVDC NSW-QLD (GW)', 'HVDC NSW-SA (GW)', 'HVDC NSW-VIC (GW)', 
        'HVDC NT-SA (GW)', 'HVDC SA-WA (GW)', 'HVDC TAS-VIC (GW)', 'LCOE ($/MWh)',
        'LCOG ($/MWh)', 'LCOB (storage)', 'LCOB (transmission)', 'LCOB (spillage & loss)']
        
    if output: ggta.to_csv('GGTA-consolidated.csv', index = False)
    
    os.chdir(active_dir)
    
    return ggta

#%%

def consolidateCapacities(scenario, relative, output=False):
    assert isinstance(scenario, int)
    assert isinstance(relative, bool)
    assert isinstance(output, bool)
    
    capacities = pd.read_csv(r'CostOptimisationResults\Optimisation_resultx{}-None.csv'.format(scenario), 
                             header = None)
    
    active_dir = os.getcwd()
    os.chdir('Results')
    
    for file in os.listdir():
        if 'Optimisation_resultx{}'.format(scenario) not in file: continue
        if str(relative) not in file: continue
        
        df = pd.DataFrame([scenario, re.search('(\[.*\]|all)', file).group()])
        df = pd.concat([df, pd.read_csv(file, header=None)], axis = 1)
        
        capacities = pd.concat([capacities, df])

    os.chdir(active_dir)

    if output: capacities.to_csv('Optimisation_resultx{}-consolidated-{}.csv'.format(scenario, relative), 
                                 index = False)

    os.chdir(active_dir)
    
    return capacities



