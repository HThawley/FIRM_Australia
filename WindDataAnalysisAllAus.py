# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:13:07 2023

@author: hmtha
"""

import os 
import pandas as pd
from tqdm import tqdm
# import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor as NestablePool
from datetime import datetime as dt
# from math import exp
# import matplotlib.pyplot as plt
# from scipy.stats import gumbel_r, poisson, pareto
# from scipy.interpolate import RBFInterpolator
import rasterio
# import geopandas as gpd
# from shapely.geometry import Point, Polygon#, MultiPolygon
# from shapely.ops import nearest_points
# from shapely import distance
# import warnings
# import geometricUtils as gmu


#%%

def readAll(location, speedThreshold, n_years, stn=None, multiprocess=False):
    active_dir = os.getcwd()
    os.chdir(location)
    
    if stn is None:
        stn = readStnDetail(location)

    folders = [(path, speedThreshold, stn, False) 
               for path in os.listdir() if ('.zip' not in path) and ('MaxWindGust' not in path)]
    
    if multiprocess:
        print('Reading and processing data: ', end = '')
        with NestablePool(max_workers=min(cpu_count(), len(folders))) as processPool:
            results = list(processPool.map(readData, folders))
        print('Completed.')
    else: 
        results = []
        for folderTuple in tqdm(folders, desc = 'Reading wind data folder-by-folder'):
            results.append(readData(folderTuple))

    coverage = sum([result[1] for result in results])
    stormDurations = pd.concat([result[0] for result in results])
    stormDurations = stormDurations.sort_values().reset_index(drop=True)
    
    stormsPerYear = len(stormDurations)/coverage 
    percentiles = [1/(stormsPerYear*N) for N in n_years]
    
    speeds = [stormSpeed(stormDurations, perc) for perc in percentiles]
    
    os.chdir(active_dir)
    return speeds, stormDurations, coverage

def stormSpeed(dist, perc): 
    return dist.tail(int(perc*len(dist))).iloc[0]

def formatStnNo(col):
    return col.apply(lambda x: '0'*(6-len(str(int(x))))+str(int(x)) if not pd.isna(x) else pd.NA)

def get_meanSpeed100m(stn):
    # The multiprocessing library doesn't play nicely with os.chdir(), hence a full path 
    try:
        windMap = rasterio.open('/media/fileshare/FIRM_Australia_Resilience/Data/AUS_wind-speed_100m.tif')
    except (FileNotFoundError, rasterio.errors.RasterioIOError):
        windMap = rasterio.open(r'C:\Users\hmtha\Desktop\FIRM_Australia\Data\AUS_wind-speed_100m.tif')

    coord_list = [(x,y) for x, y in zip(stn['longitude'], stn['latitude'])]

    stn['meanSpeed-100m'] = [x[0] for x in windMap.sample(coord_list)]
    stn['meanSpeed-100m'] = pd.to_numeric(stn['meanSpeed-100m'])

    windMap.close()   
    return stn

def readStnDetail(location):
    active_dir = os.getcwd()
    os.chdir(location)
    
    loc_dir = os.getcwd()
    
    folders = [folder for folder in os.listdir() if ('.zip' not in folder) and ('MaxWindGust' not in folder)]

    stn = pd.DataFrame([])
    for folder in folders:
        os.chdir(folder)
        stnDet = [path for path in os.listdir() if 'StnDet' in path]
        assert len(stnDet) == 1
        
        stn = pd.concat([stn, pd.read_csv(stnDet[0], header = None, usecols = [1,3,6,7,10], dtype=str)])
        
        os.chdir(loc_dir)
        
    stn.columns = ['station no.', 'station name', 'latitude', 'longitude', 'altitude']
    stn['station no.'] = formatStnNo(stn['station no.'])
    for col in ('latitude', 'longitude', 'altitude'):
        stn[col] = pd.to_numeric(stn[col])
            
    stn = get_meanSpeed100m(stn)
    os.chdir(active_dir)
    return stn

def readData(argTuple, folder=None, speedThreshold=None, stn=None, multiprocess=None):
    if argTuple is not None: 
        folder, speedThreshold, stn, multiprocess = argTuple
    for arg in (folder, speedThreshold, stn, multiprocess):
        assert arg is not None
    active_dir = os.getcwd()
    os.chdir(folder)
    
    argTuples = [(path, speedThreshold, stn) for path in os.listdir() if 'Data' in path]
    
    if multiprocess:
        with Pool(processes = min(cpu_count(), len(argTuples))) as processPool:
            resultGen = processPool.map(readFile, argTuples)
            results = list(resultGen)

    else: 
        results = [readFile(argTuple) for argTuple in argTuples]
    
    coverage = sum([result[1] for result in results])
    stormDurations = pd.concat([result[0] for result in results])

    os.chdir(active_dir)
    return stormDurations, coverage

    
def readFile(argTuple=None, path=None, speedThreshold=None, stn=None):
    if argTuple is not None: 
        path, speedThreshold, stn = argTuple
    for arg in (path, stn, speedThreshold):
        assert arg is not None

    stnNo = path[11:17]
    if stnNo not in set(stn['station no.']):
        return pd.Series([]), 0
    
    df = pd.read_csv(path, usecols = [7,8,9,10,11,12,16], dtype = str)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')  
    df = df.rename(columns={'Speed of maximum windgust in last 10 minutes in  km/h':'gustSpeed-10m', 
                          'Wind speed in km/h':'meanSpeed-10m'})
    
    df['dt'] = df[df.columns[0:5]].apply(lambda row: lambdaDt(*row), axis=1)
    
    df['gustSpeed-10m'] = df['gustSpeed-10m'] / 3.6# km/h to m/s
    df['meanSpeed-10m'] = df['meanSpeed-10m'] / 3.6# km/h to m/s
    df = df[['dt', 'gustSpeed-10m', 'meanSpeed-10m']]

    if len(df.dropna()) == 0: 
        return pd.Series([]), 0
    
    df = df.dropna(subset=['gustSpeed-10m'])
    df = df.sort_values(by=['dt', 'gustSpeed-10m'])
    df = df.drop_duplicates(subset='dt', keep='last')
    
    longTermMeanSpeed = df['meanSpeed-10m'].mean()    
    startTime, endTime = df.dropna(subset=['gustSpeed-10m'])['dt'].min(), df.dropna(subset=['gustSpeed-10m'])['dt'].max()
    # meanRes = (df.dropna(subset=['gustSpeed-10m'])[['dt']].diff(periods=1, axis=0).sum()[0]/len(df)).total_seconds()/60
    
    try: 
        scaleFactor = stn[stn['station no.']==stnNo]['meanSpeed-100m'].values[0] / longTermMeanSpeed
    except KeyError: 
        return pd.Series([]), 0
    if pd.isna(scaleFactor): 
        return pd.Series([]), 0

    df['gustSpeed-100m'] = scaleFactor * df['gustSpeed-10m']

    #define storm indicator initially
    df['highSpeedInd'] = df['gustSpeed-100m'].apply(lambda gust: 2 if gust >=speedThreshold[0] else 1 if gust>=speedThreshold[1] else 0)
    #smooth storm indicator to include adjacent instances within 10% of cutOffSpeed
    df['highSpeedInd'] = df['highSpeedInd'].rolling(window=2).mean().apply(lambda ind: 1 if ind>=1.5 else 0)
    # Calculate time between rows 
    df['timeDiff'] = (df['dt'].shift(-1) - df['dt']).apply(lambda td: td.days*24 + td.seconds/3600)
    df.iloc[-1, df.columns.get_indexer(['timeDiff'])] = df['timeDiff'].mean()

    # Where a time period is greater than 2 hours, do not let it be a storm
    # This prevents some incorrect 600+ hour storms 
    df.loc[df['timeDiff'] > 2, 'highSpeedInd'] = 0 
    
    # Calculate the length of time periods either or not experiencing a storm
    periodDurations = df['timeDiff'].groupby((df['highSpeedInd'] != df['highSpeedInd'].shift()).cumsum()).sum()
   
    # Mask for which time periods are storms 
    periodIndicators = df['highSpeedInd'].groupby((df['highSpeedInd'] != df['highSpeedInd'].shift()).cumsum()).unique().astype(bool)
    stormDurations = periodDurations[periodIndicators]

    # Exclude periods of time with insufficient coverage, 
    timeSpan = endTime-startTime
    coverage = (((timeSpan.seconds/86400.) + timeSpan.days)/365.25) - ((df['timeDiff'][df['timeDiff']>2].sum())/8766)

    return stormDurations, coverage

def lambdaDt(y, mo, d, h, mi): return dt(y, mo, d, h, mi)
    
#%%
def removeAnomalousStns(stn):
    badStns = ['053000','041560','068076','030024','016092','023849','018207', 
               '092037','094250','091375','092133','092163','087185','078072','078031']
    return stn[~stn['station no.'].isin(badStns)]

#%%
 
if __name__=='__main__':

    speedThreshold=(25, #turbine cut-off speed 25 m/s
                    25*0.9 #wind gust speed tolerance, 10% 
                    )
    
    n_years=(5,10,20,25,50,100)
    
    stn = readStnDetail(r'BOM Wind Data')
    
    speeds, stormDurations, coverage = readAll(r'BOM Wind Data', speedThreshold, n_years, stn=stn, multiprocess=True)
    
    stormDurations.value_counts().to_csv('Data/stormDurations.csv', header=False)
    pd.DataFrame([coverage]).to_csv('Data/stormCoverage.csv', index=False, header=False)

    # stn.to_csv(r'Data/WindStats.csv', index = False)
    
    # stn = pd.read_csv(r'Data/WindStats.csv')
    # stn['station no.'] = formatStnNo(stn['station no.'])

    # stn = findClosestZones(stn, 50) #km
    # stn = stn.dropna(subset=['highWindFrac', 'meanDuration'], how='any')

    # stn = removeAnomalousStns(stn)

    # zones = zoneAnalysis(stn, plot = False).sort_values('zone').reset_index(drop=True) 
    # zones.to_csv('Results/windDataByZone/_zoneData.csv', index=False)
    
    # # plotMap(stn)     

    # #Manual Analysis
    # # grpby = stn.groupby('closestZone')[['meanSpeed-10m','meanSpeed-100m','meanDuration',
    # #     'highWindFrac','scaleFactor','meanRes','Observations']].describe()
    # # grpby = grpby.drop(columns=[col for col in grpby.columns if '%' in col[1] \
    # #         or 'std' in col[1] or 'count'==col[1] and 'meanSpeed-10m'!=col[0]])
    # # grpby.to_csv('Results/zoneWindStats.csv')
    
    # for i, df in stn.groupby('closestZone'):
    #     df.to_csv(f'Results/windDataByZone/Zone{i}.csv', index=False)


