# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:50:04 2023

@author: hmtha
"""

import os 
import pandas as pd
from tqdm import tqdm
from math import sin, cos, sqrt, atan2, pi
import numpy as np

#%%
def readData(repo):
    os.chdir(repo)
    # z = pd.Series([])
    
    stn = pd.read_csv('C:/Users/hmtha/OneDrive/Desktop/data - Copy/AWS_Wind-NT/HM01X_StnDet_9999999910323091.txt',
                      header = None,
                      usecols = [1,3,6,7])
    stn.columns = ['stn no.', 'stn Name', 'Lat', 'Lon']
    
    stnData = {}
    stnFracs = {}
    
    # files = os.listdir()
    for f in tqdm(os.listdir()):
        if not 'Data' in f:
            continue
        stnNo = f[11:17]
        
        x = pd.read_csv(f, usecols = [16],dtype=str)
        gusts = pd.to_numeric(x['Speed of maximum windgust in last 10 minutes in  km/h'],
                              errors = 'coerce').dropna()
        gusts = gusts / 3.6# km/h to m/s
        
        frac = len(gusts[gusts > 25])/len(gusts)
        
        # z = pd.concat([z, gusts])
        stnData[stnNo] = gusts
        stnFracs[stnNo] = frac
    
    stn['frac'] = stn['stn no.'].apply(lambda x: '0'+str(x) 
                                       if len(str(x)) == 5 
                                       else str(x)
                                       ).map(stnFracs)
    return stn

#%%
def fracFromCoord(lat, lon, stn, k='max'):
    '''
    Determine frac as a weighted average of k-nearest neighbours.
    
    lat and lon are coordinates (in degrees) where frac to be found
    
    stn is dataframe of stn details 
    '''

    if k=='max':
        k=len(stn)
        
    assert k<=len(stn)
    
    stnCopy = stn.copy()
    
    stnCopy['distance'] = stnCopy[['Lat', 'Lon']].apply(lambda x: Haversine(*x, lat, lon), axis = 1)
    
    stnCopy = stnCopy.sort_values('distance', ascending = True).reset_index(drop=True)

    stnCopy = stnCopy.iloc[:k]

    stnCopy['weightedFrac'] = stnCopy['frac'] / stnCopy['distance']
    
    
    return stnCopy['weightedFrac'].sum()/((1/stnCopy['distance']).sum())

def Haversine(lat1,lon1,lat2,lon2, **kwarg):
    """
    This uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is, 
    the shortest distance over the earth’s surface – giving an ‘as-the-crow-flies’ distance between the points 
    (ignoring any hills they fly over, of course!).
    Haversine
    formula:    a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    c = 2 ⋅ atan2( √a, √(1−a) )
    d = R ⋅ c
    where   φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
    note that angles need to be in radians to pass to trig functions!
    """
    R = 6371.0088
    lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = R * c
    return d
 
stn = readData(r'C:\Users\hmtha\OneDrive\Desktop\data - Copy\AWS_Wind-NT')   
x = fracFromCoord(-12, 130, stn, 'max')
    
    
    
    
