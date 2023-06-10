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
from datetime import datetime as dt
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
import rasterio


#%%

def readAll(location):
    active_dir = os.getcwd()
    os.chdir(location)
    
    folders = [path for path in os.listdir() if ('.zip' not in path) and ('MaxWindGust' not in path)]
    
    print('Reading and processing data\nCompleted: ', end = '')
    pool = NestablePool(max_workers=min(cpu_count(), len(folders)))
    results = pool.map(readData, folders)
    
    stn = pd.concat(results, ignore_index=True)
    
    os.chdir(active_dir)
    
    return stn

def readStnDetail(folder):
    
    stnDet = [path for path in os.listdir() if 'StnDet' in path]
    
    assert len(stnDet) == 1
    
    stn = pd.read_csv(stnDet[0],
                      header = None,
                      usecols = [1,3,6,7])
    stn.columns = ['station no.', 'station name', 'latitude', 'longitude']
    
    stn['station no.'] = stn['station no.'].apply(lambda x: '0'*(6-len(str(x)))+str(x) 
                                          if not pd.isna(x)
                                          else pd.NA)    

    stn['latitude'] = pd.to_numeric(stn['latitude'])
    stn['longitude'] = pd.to_numeric(stn['longitude'])
    return stn

def get100mMeans(stn):
    windMap = rasterio.open('/media/fileshare/FIRM_Australia_Resilience/Data/AUS_wind-speed_100m.tif')

    coord_list = [(x,y) for x, y in zip(stn['longitude'], stn['latitude'])]

    stn['mean-100m'] = [x[0] for x in windMap.sample(coord_list)]

    windMap.close()   
    return stn

def readData(folder, multiprocess=True):

    active_dir = os.getcwd()
    os.chdir(folder)
    
    stn = readStnDetail(folder)
    stn = get100mMeans(stn)
    
    argTuples = [(path, stn) for path in os.listdir() if 'Data' in path]

    # result=[]
    # if multiprocess:
    #     with Pool(processes = min(cpu_count(), len(argTuples))) as p:
    #         with tqdm(total=len(argTuples), desc=f"Processing: {folder}") as pbar:
    #             for inst in p.imap_unordered(readFile, argTuples):
    #                 pbar.update()
    #                 result.append(inst)
    # else: 
    #     for argTuple in tqdm(argTuples):
    #         result.append(readFile(argTuple))
    
    result=[]
    if multiprocess:
        with Pool(processes = min(cpu_count(), len(argTuples))) as p:
            for inst in p.imap_unordered(readFile, argTuples):
                result.append(inst)
    else: 
        for argTuple in argTuples:
            result.append(readFile(argTuple))
            
    result = pd.DataFrame(result, 
                          columns = ['station no.', 'highWindFrac', 'scaleFactor',
                                     'mean-10m', 'startTime', 'meanRes', 'Observations'])
    stn = stn.merge(result, on = 'station no.', how = 'outer', indicator ='indicator')
    
    assert len(stn['indicator'].unique()) == 1 
    assert stn['indicator'].unique()[0] == 'both'
    
    stn = stn.drop(columns = 'indicator')
    
    print(f'{folder[9:]}; ', end='')
    os.chdir(active_dir)
    return stn    
    
def readFile(argTuple):
        
    path, stn = argTuple
    x = pd.read_csv(path, usecols = [2,3,4,5,6,16], dtype = str)
    
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors = 'coerce')
        
    x = x.rename(columns={'Speed of maximum windgust in last 10 minutes in  km/h':'speed'})
    
    x['dt'] = x[x.columns[0:5]].apply(lambda row: lambdaDt(*row), axis=1)
    
    # x['speed'] = pd.to_numeric(x['speed'], errors = 'coerce')
    x['speed'] = x['speed'] / 3.6# km/h to m/s
    
    x = x[['dt', 'speed']].dropna()    
    
    if len(x) == 0:
        return [path[11:17], pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]
    
    stnNo = path[11:17]
    
    try: 
        scaleFactor = stn[stn['station no.']==stnNo]['mean-100m'].values[0] / x['speed'].mean()
    except KeyError: 
        return [path[11:17], pd.NA, pd.NA, x['speed'].mean(), x['dt'].min(), (x[['dt']].diff(periods=1, axis=0).sum()[0]/len(x)).total_seconds()/60, len(x)]
    if pd.isna(scaleFactor):
        return [path[11:17], pd.NA, pd.NA, x['speed'].mean(), x['dt'].min(), (x[['dt']].diff(periods=1, axis=0).sum()[0]/len(x)).total_seconds()/60, len(x)]
    
    x['speed 100m high res'] = scaleFactor * x['speed']
    
    highWindIntegral = gumbelModelling(x['speed 100m high res'])
    meanSpeed = x['speed'].mean()
    
    
    startTime = x['dt'].min()
    meanRes = (x[['dt']].diff(periods=1, axis=0).sum()[0]/len(x)).total_seconds()/60
    
    return [path[11:17], highWindIntegral, scaleFactor, meanSpeed, startTime, meanRes, len(x)]

def lambdaRollDiff(a,b): return a-b

def lambdaDt(y, mo, d, h, mi): return dt(y, mo, d, h, mi)

def gumbelModelling(dist, plot=False):    
    a, c = gumbel_r.fit(dist)
        
    if plot:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax2.hist(dist, bins=50, alpha = 0.5)
        
        x = np.linspace(gumbel_r.ppf(0.000001, a, c),
                    gumbel_r.ppf(0.999999, a, c), 200)
        ax1.plot(x, gumbel_r.pdf(x, a, c),
           'r-', lw=2, alpha=0.9, label='gumbel pdf')
    
    return 1 - gumbel_r.cdf(25, a, c)


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
    lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = 6371.0088 * c
    return d
 
#%%    
 
if __name__=='__main__':
 
    stn = readAll(r'BOM Wind Data')
    # stn = readData(r'BOM Wind Data\AWS_Wind-NT', multiprocess=True)   



    stn.to_csv(r'Data/WindStats.csv', index = False)
#%%





