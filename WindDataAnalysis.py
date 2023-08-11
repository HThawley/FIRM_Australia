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
# from math import exp
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, poisson, pareto
from scipy.interpolate import RBFInterpolator
import rasterio
import geopandas as gpd
from shapely.geometry import Point, Polygon#, MultiPolygon
from shapely.ops import nearest_points
# from shapely import distance
import warnings


import geometricUtils as gmu


#%%

def readAll(location, speedThreshold, n_years, stn=None, multiprocess=False):
    active_dir = os.getcwd()
    os.chdir(location)
    
    if stn is None:
        stn = readStnDetail(location)

    folders = [(path, stn, speedThreshold, n_years, False) 
               for path in os.listdir() if ('.zip' not in path) and ('MaxWindGust' not in path)]
    
    if multiprocess:
        print('Reading and processing data\nCompleted: ', end = '')
        with NestablePool(max_workers=min(cpu_count(), len(folders))) as processPool:
            results = processPool.map(readData, folders)
        print('')
    else: 
        results = []
        for folderTuple in tqdm(folders, desc = 'Reading wind data folder-by-folder'):
            results.append(readData(folderTuple))

    stn = pd.concat(results, ignore_index=True)
    
    os.chdir(active_dir)
    return stn

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

def readData(argTuple, folder=None, stn=None, speedThreshold=None, n_years=None, multiprocess=None):
    if argTuple is not None: 
        folder, stn, speedThreshold, n_years, multiprocess = argTuple
    for arg in (folder, stn, speedThreshold, multiprocess):
        assert arg is not None
        
    active_dir = os.getcwd()
    os.chdir(folder)
    
    argTuples = [(path, stn, speedThreshold, n_years) for path in os.listdir() if 'Data' in path]
    

    if multiprocess:
        with Pool(processes = min(cpu_count(), len(argTuples))) as processPool:
            result = processPool.map(readFile, argTuples)

    else: 
        result=[readFile(argTuple) for argTuple in argTuples]
    
    result = pd.DataFrame(result, columns = ['station no.', 'scaleFactor', 'meanSpeed-10m', 
                                             'startTime', 'meanRes', 'Observations', 'Storm no.']
                          +[f'1-in-{N}-year duration' for N in n_years])
    
    stn = stn.dropna(subset = 'station no.')
    stn = stn.merge(result, on = 'station no.', how = 'inner', indicator ='indicator')
    
    assert len(stn['indicator'].unique()) == 1
    assert stn['indicator'].unique()[0] == 'both'
    
    stn = stn.drop(columns = 'indicator')
    
    print(f'{folder[9:]}; ', end='')
    os.chdir(active_dir)
    return stn    
    
def readFile(argTuple=None, path=None, stn=None, speedThreshold=None, n_years=None):
    if argTuple is not None: 
        path, stn, speedThreshold, n_years = argTuple
    for arg in (path, stn, speedThreshold):
        assert arg is not None
            
    stnNo = path[11:17]
    if stnNo not in set(stn['station no.']):
        return [pd.NA] * (7+ len(n_years))
    
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
        return [stnNo] + [pd.NA] * (6 + len(n_years))
    
    df = df.dropna(subset=['gustSpeed-10m'])
    df = df.sort_values(by=['dt', 'gustSpeed-10m'])
    df = df.drop_duplicates(subset='dt', keep='last')
    
    longTermMeanSpeed = df['meanSpeed-10m'].mean()    
    startTime, endTime = df.dropna(subset=['gustSpeed-10m'])['dt'].min(), df.dropna(subset=['gustSpeed-10m'])['dt'].max()
    meanRes = (df.dropna(subset=['gustSpeed-10m'])[['dt']].diff(periods=1, axis=0).sum()[0]/len(df)).total_seconds()/60
    
    try: 
        scaleFactor = stn[stn['station no.']==stnNo]['meanSpeed-100m'].values[0] / longTermMeanSpeed
    except KeyError: 
        return [stnNo, pd.NA, longTermMeanSpeed, startTime, meanRes, len(df), pd.NA] + [pd.NA] * len(n_years)
    if pd.isna(scaleFactor): 
        return [stnNo, pd.NA, longTermMeanSpeed, startTime, meanRes, len(df), pd.NA] + [pd.NA] * len(n_years)

    df['gustSpeed-100m'] = scaleFactor * df['gustSpeed-10m']

    #define storm indicator initially
    df['highSpeedInd'] = df['gustSpeed-100m'].apply(lambda gust: 2 if gust >=speedThreshold[0] else 1 if gust>=speedThreshold[1] else 0)
    #smooth storm indicator to include adjacent instances within 10% of cutOffSpeed
    df['highSpeedInd'] = df['highSpeedInd'].rolling(window=2).mean().apply(lambda ind: 1 if ind>=1.5 else 0)
    # Calculate time between rows 
    df['timeDiff'] = (df['dt'].shift(-1) - df['dt']).apply(lambda td: td.days*24 + td.seconds/3600)
    df['timeDiff'].iloc[-1] = df['timeDiff'].mean()

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

    try: 
        stormsPerYear = len(stormDurations)/coverage
        durations = paretoModelling(stormDurations, n_years, stormsPerYear)
    except ZeroDivisionError:
        durations = [0 for N in n_years]
    
    return [stnNo, scaleFactor, longTermMeanSpeed, startTime, meanRes, len(df.dropna()), len(stormDurations)] + durations

def lambdaDt(y, mo, d, h, mi): return dt(y, mo, d, h, mi)

#%%

def poissonModelling(data, n_years, stormsPerYear):
    
    mean = data.mean()
   
    try: return [poisson.ppf((1-1/(N*stormsPerYear)), mean) for N in n_years]
    
    except ZeroDivisionError: return [0 for N in n_years]
    
def paretoModelling(data, n_years, stormsPerYear):
    
    try: p = pareto.fit(data)
    except ValueError: return [0 for N in n_years]
    
    try: return [pareto.ppf((1-1/(N*stormsPerYear)), *p) for N in n_years]
    except ZeroDivisionError: return [0 for N in n_years]
    
    
    
    
    
#%%
def lambdaDistanceEdge(point, polygon): 
    p1, p2 = nearest_points(polygon, point)
    return gmu.Haversine(p1.y, p1.x, p2.y, p2.x)

def lambdaDistancePoints(p1, p2):
    return gmu.Haversine(p1.x, p1.y, p2.x, p2.y)

def lambdaDuplicateZones(old, new):
    if isinstance(old, list):
        old.append(new)
        return old
    else: 
        return [old, new]
    
def findClosestZones(stn, distanceThreshold):

    stn['point'] = stn[['longitude', 'latitude']].apply(lambda coord: Point(*coord), axis = 1)
    
    stn['closestZone'] = pd.NA
    stn['distanceToZone'] = np.inf
    stn['distanceToCentroid'] = np.inf
    
    active_dir = os.getcwd()
    os.chdir('Geometries/wind')
    
    for zone in os.listdir():
        poly = gpd.read_file(zone)['geometry'][0]
        distances = stn['point'].apply(lambda point: lambdaDistanceEdge(point, poly))
        centroidDistance = stn['point'].apply(lambda point: lambdaDistancePoints(point, poly.centroid))
        inZone = stn['point'].apply(lambda point: poly.contains(point))
        
        distanceMask = (distances < stn['distanceToZone'])
        
        #a couple of zones are duplicated
        duplicateMask = (distances == stn['distanceToZone'])
        
        stn.loc[distanceMask,'distanceToZone'] = distances[distanceMask]
        stn.loc[distanceMask,'distanceToCentroid'] = centroidDistance[distanceMask]
        stn.loc[distanceMask,'inZone'] = inZone[distanceMask]
        
        stn.loc[duplicateMask, 'closestZone'] = stn.loc[duplicateMask, 'closestZone'].apply(
            lambda firstZone: lambdaDuplicateZones(firstZone, zone.split('.')[0]))
        
        stn.loc[distanceMask,'closestZone'] = zone.split('.')[0]
        
    stn = stn.explode('closestZone').reset_index(drop=True)
    stn['closestZone'] = pd.to_numeric(stn['closestZone'])
    stn = stn[stn['distanceToZone'] < distanceThreshold]    
    stn = stn.drop(columns=['point'])
    os.chdir(active_dir)
    
    return stn

#%%

def interpolate(stn, zone, poly):
    stn = stn[stn['closestZone'] == zone]
    
    coords = np.array(list(zip(stn['longitude'], stn['latitude'])))
    highWindFrac = np.array(stn['highWindFrac'])
    meanDuration = np.array(stn['meanDuration'])
    centroid = poly['geometry'][0].centroid.x, poly['geometry'][0].centroid.y
    
    hwf = RBFInterpolator(coords, highWindFrac, kernel = 'linear', degree = 0)
    mdr = RBFInterpolator(coords, meanDuration, kernel = 'linear', degree = 0)
    
    highWindFrac = hwf(np.array([centroid]))
    meanDuration = mdr(np.array([centroid]))
        
    zoneDf = pd.DataFrame(
        [[stn['closestZone'].unique()[0], 
         highWindFrac[0], 
         meanDuration[0]]], 
        columns = ['zone', 'highWindFrac', 'meanDuration'])
    
    return zoneDf

def zoneAnalysis(stn, plot = False):
    active_dir = os.getcwd()
    os.chdir('Geometries/wind')
    zoneFiles = os.listdir()
    
    os.chdir(active_dir)
    
    zoneDf = pd.DataFrame([])    
    for zone in zoneFiles:
        poly = gpd.read_file('Geometries/wind/'+zone)
        zone = int(zone.split('.')[0])
        
        zoneDf = pd.concat([zoneDf, interpolate(stn, zone, poly, plot)])
    
    os.chdir(active_dir)
    return zoneDf

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
    stn = findClosestZones(stn, 50)
    
    stn = readAll(r'BOM Wind Data', speedThreshold, n_years, stn=stn, multiprocess=True)
    stn.to_csv(r'Results/statModels/ParetoWindStats1.csv')


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


