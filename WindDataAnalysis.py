# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:50:04 2023

@author: hmtha
"""

import os 
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor as NestablePool



#%%

def readAllRepo(location):
    active_dir = os.getcwd()
    
    os.chdir(location)
    repos = [path for path in os.listdir() if '.zip' not in path]
    
    print('Reading and processing data\nCompleted: ', end = '')
    pool = NestablePool(max_workers=min(cpu_count(), len(repos)))
    results = pool.map(readData, repos)
    
    stn = pd.concat(results, ignore_index=True)
    
    os.chdir(active_dir)
    
    return stn

def readData(repo):
    active_dir = os.getcwd()
    os.chdir(repo)
    
    print(f'{repo[9:]} ', end='')
    
    stn = pd.read_csv('C:/Users/hmtha/OneDrive/Desktop/data - Copy/AWS_Wind-NT/HM01X_StnDet_9999999910323091.txt',
                      header = None,
                      usecols = [1,3,6,7])
    stn.columns = ['stn no.', 'stn Name', 'Lat', 'Lon']
    
    stn['stn no.'] = stn['stn no.'].apply(lambda x: '0'*(6-len(str(x)))+str(x) 
                                          if not pd.isna(x)
                                          else pd.NA)
    
    files = [path for path in os.listdir() if 'Data' in path]
    
    # pool = Pool(processes = min(cpu_count(), len(files)))
    # result = pool.map(readFile, files)
    # pool.terminate()
    result = []
    for path in tqdm(files):
        result.append(readFile(path))
    
    stn['frac'] = stn['stn no.'].map(dict(zip([output[0] for output in result], 
                                              [output[1] for output in result])))
    stn['gust'] = stn['stn no.'].map(dict(zip([output[0]for output in result], 
                                              [output[2] for output in result])))
    
    os.chdir(active_dir)
    return stn    
    
def readFile(path):
    
    x = pd.read_csv(path, usecols = [16],dtype=str)
    gust = pd.to_numeric(x['Speed of maximum windgust in last 10 minutes in  km/h'],
                          errors = 'coerce').dropna()
    gust = gust / 3.6# km/h to m/s
    
    frac = len(gust[gust > 25])/len(gust) if len(gust) > 0 else 0 

    return (path[11:17], frac, gust)

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

def Haversine(lat1,lon1,lat2,lon2):
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
 
if __name__=='__main__':
    stn = readAllRepo(r'C:\Users\hmtha\OneDrive\Desktop\data - Copy')
    # stn = readData(r'C:\Users\hmtha\OneDrive\Desktop\data - Copy\AWS_Wind-NT')   
    # x = fracFromCoord(-12, 130, stn, 'max')
    
#%%

from math import exp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def weibull(x, c, k, loc, scale): 
    if x <0:
        return 0 
    return ((k/c)*(((x-loc)/c)**(k-1))*(exp(-((x-loc)/c)**k)))/scale

def weibullIntegral(a,b,l,k):
    return exp(-(a/l)**k)-exp(-(b/l)**k)
    
def weibullNormaliseDistribution(dist):
    return stats.exponweib.fit(dist)
    

xrange = np.arange(0,30000)/1000.
yrange = [weibull(x, 12, 1.5, 0, 2.) for x in xrange]
yrange2 = [weibull(x, 10, 1.5, 0, 2.) for x in xrange]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.set_ylabel("Frequency")
ax.set_xlabel("Wind speed  (m/s)")
ax.set_title("Wind speed distribution")

plt.plot(xrange, yrange, color = 'red')
plt.plot(xrange, yrange2, color = 'blue') 

s = np.arange(25000,30000)/1000

plt.fill_between(x= xrange, y1= yrange, where= xrange > 25,color= "red",alpha= 0.2)
plt.fill_between(x= xrange, y1= yrange2, where= xrange > 25,color= "blue",alpha= 0.2)

plt.plot([25, 25], [-0.001, max(max(yrange), max(yrange2))*1.1], color = 'green')
plt.plot([0, 30], [0,0], color = 'black')

