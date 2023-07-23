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
from scipy.stats import gumbel_r
from scipy.interpolate import RBFInterpolator#, LinearNDInterpolator
import rasterio
import geopandas as gpd
from shapely.geometry import Point, Polygon#, MultiPolygon
from shapely.ops import nearest_points
# from shapely import distance
import warnings
from scipy.stats import exponweib, weibull_min, expon, genextreme, goodness_of_fit, weibull_max

import geometricUtils as gmu


#%%

def readAll(location, speedThreshold, multiprocess=False, goodnessOfFit=None):
    active_dir = os.getcwd()
    os.chdir(location)
    
    folders = [(path, speedThreshold, multiprocess, goodnessOfFit) for path in os.listdir() if ('.zip' not in path) and ('MaxWindGust' not in path)]
    
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

def readStnDetail(folder):
    
    stnDet = [path for path in os.listdir() if 'StnDet' in path]
    assert len(stnDet) == 1
    
    stn = pd.read_csv(stnDet[0], header = None, usecols = [1,3,6,7,10])
    stn.columns = ['station no.', 'station name', 'latitude', 'longitude', 'altitude']
    
    stn['station no.'] = formatStnNo(stn['station no.'])
    
    stn['latitude'] = pd.to_numeric(stn['latitude'])
    stn['longitude'] = pd.to_numeric(stn['longitude'])
    return stn

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

def readData(argTuple, folder=None, speedThreshold=None, multiprocess=True, goodnessOfFit=None):
    if argTuple is not None: 
        folder, speedThreshold, multiprocess, goodnessOfFit = argTuple
    for arg in (folder, speedThreshold, multiprocess):
        assert arg is not None
        
    active_dir = os.getcwd()
    os.chdir(folder)
    
    stn = readStnDetail(folder)
    stn = get_meanSpeed100m(stn)
    
    argTuples = [(path, stn, speedThreshold, goodnessOfFit) for path in os.listdir() if 'Data' in path]
    
    result=[]
    if multiprocess:
        with Pool(processes = 4#min(cpu_count(), len(argTuples))) 
        )as processPool:
            for inst in processPool.imap_unordered(readFile, argTuples):
                result.append(inst)
    else: 
        for argTuple in argTuples:
            result.append(readFile(argTuple))
    
    if goodnessOfFit is not None: 
        result = pd.DataFrame(result, columns = ['station no.', 'gustObs', 'stormCount', 'startTime', 
                                                 'meanRes', 'gustGoodness', 'durationGoodness'])
    else:
        result = pd.DataFrame(result, columns = ['station no.', 'meanDuration', 'highWindFrac', 'scaleFactor',
                                                 'meanSpeed-10m', 'startTime', 'meanRes', 'Observations'])
    stn = stn.merge(result, on = 'station no.', how = 'outer', indicator ='indicator')
    
    assert len(stn['indicator'].unique()) == 1 
    assert stn['indicator'].unique()[0] == 'both'
    
    stn = stn.drop(columns = 'indicator')
    
    print(f'{folder[9:]}; ', end='')
    os.chdir(active_dir)
    return stn    
    
def readFile(argTuple, path=None, stn=None, speedThreshold=None, goodnessOfFit=None):
    if argTuple is not None: 
        path, stn, speedThreshold, goodnessOfFit = argTuple
    for arg in (path, stn, speedThreshold):
        assert arg is not None
            
    stnNo = path[11:17]
    df = pd.read_csv(path, usecols = [2,3,4,5,6,12,16], dtype = str)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')  
    df = df.rename(columns={'Speed of maximum windgust in last 10 minutes in  km/h':'gustSpeed-10m', 
                          'Wind speed in km/h':'meanSpeed-10m'})
    
    df['dt'] = df[df.columns[0:5]].apply(lambda row: lambdaDt(*row), axis=1)
    
    df['gustSpeed-10m'] = df['gustSpeed-10m'] / 3.6# km/h to m/s
    df['meanSpeed-10m'] = df['meanSpeed-10m'] / 3.6# km/h to m/s
    df = df[['dt', 'gustSpeed-10m', 'meanSpeed-10m']]

    if len(df.dropna()) == 0: 
        return readFileBadReturn([stnNo, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA], goodnessOfFit, stnNo)
    
    longTermMeanSpeed = df['meanSpeed-10m'].dropna().mean()
    startTime = df.dropna(subset=['gustSpeed-10m'])['dt'].min()
    meanRes = (df.dropna(subset=['gustSpeed-10m'])[['dt']].diff(periods=1, axis=0).sum()[0]/len(df)).total_seconds()/60
    
    try: 
        scaleFactor = stn[stn['station no.']==stnNo]['meanSpeed-100m'].values[0] / longTermMeanSpeed
    except KeyError: 
        return readFileBadReturn([stnNo, pd.NA, pd.NA, pd.NA, longTermMeanSpeed, startTime, meanRes, len(df)], goodnessOfFit, stnNo)
    if pd.isna(scaleFactor): 
        return readFileBadReturn([stnNo, pd.NA, pd.NA, pd.NA, longTermMeanSpeed, startTime, meanRes, len(df)], goodnessOfFit, stnNo)
    
    df['gustSpeed-100m'] = scaleFactor * df['gustSpeed-10m']
   
    #define storm indicator initially
    df['highSpeedInd'] = df['gustSpeed-100m'].apply(lambda gust: 2 if gust >=speedThreshold[0] else 1 if gust>=speedThreshold[1] else 0)
    #smooth storm indicator to include adjacent instances within 10% of cutOffSpeed
    df['highSpeedInd'] = df['highSpeedInd'].rolling(window=2).mean().apply(lambda ind: 1 if ind>=1.5 else 0)
    #https://stackoverflow.com/questions/37934399/identifying-consecutive-occurrences-of-a-value-in-a-column-of-a-pandas-dataframe
    df['duration'] = df['highSpeedInd'].groupby((df['highSpeedInd'] != df['highSpeedInd'].shift()).cumsum()).transform('size') * df['highSpeedInd']
         
    lengths = df['duration'].value_counts()
    #exclude storms 1.5 hours or less in duration
    for len_excl in (0,1):#,1,2,3):
        try: lengths = lengths.drop(index=len_excl)    
        except KeyError: pass
    lengths = lengths.reset_index().rename(columns={0:'count', 'index':'count'})
    mask = lengths['duration']>0
    lengths.loc[mask, 'count'] = lengths.loc[mask, 'count']/lengths.loc[mask,'duration']
    
    if goodnessOfFit is not None: 
        highWindGoodness, durationGoodness = goodnessOfFitTesting(goodnessOfFit, df, lengths)
        try: gustObs = len(df['gustSpeed-100m'].dropna())
        except: gustObs = pd.NA
        try: durObs = lengths['count'].sum()
        except: durObs = pd.NA
        return [stnNo, gustObs, durObs, startTime, meanRes, highWindGoodness, durationGoodness]
        
    else: 
        highWindIntegral, meanDuration = statisticalAnalysis(df, lengths, speedThreshold, stnNo)
    
    return [stnNo, meanDuration, highWindIntegral, scaleFactor, longTermMeanSpeed, startTime, meanRes, len(df.dropna())]

def readFileBadReturn(dfRow, distribution, stnNo):
    if distribution is not None: return [stnNo, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]
    else: return dfRow

def goodnessOfFitTesting(distribution, df, lengths):    
    try: 
        highWindGoodness = goodness_of_fit(distribution, df['gustSpeed-100m'].dropna(), n_mc_samples=100)[2] #pvalue
    except: 
        highWindGoodness = pd.NA
    try: 
        durationGoodness = goodness_of_fit(distribution, np.repeat(lengths['duration'], lengths['count']), n_mc_samples=100)[2]
    except: 
        durationGoodness = pd.NA
    return highWindGoodness, durationGoodness

def statisticalAnalysis(df, lengths, speedThreshold, stnNo=None):
   
    highWindIntegral = gumbelModelling(df['gustSpeed-100m'].dropna(), 
                                       cutOffSpeed=speedThreshold[0], 
                                       plot=False)#, stnNo = stnNo, data = 'gust')
    
    # duration = gumbelModelling(np.repeat(lengths['duration'], lengths['count']), 
    #                            percentile = 0.75, plot=True, stnNo = stnNo, data = 'durations')

    try: 
        meanDuration = lengths.prod(axis=1).sum()/lengths['count'].sum()
    except ZeroDivisionError: 
        meanDuration = 0
    meanDuration = 0 if pd.isna(meanDuration) else meanDuration
        
    return highWindIntegral, meanDuration

def lambdaDt(y, mo, d, h, mi): return dt(y, mo, d, h, mi)

#%%

def gumbelModelling(dist, cutOffSpeed=None, percentile=None, plot=False, 
                    scale='log', stnNo = None, data=None): 
    n = gumbel_r.fit(dist)
        
    if plot:
        p = exponweib.fit(dist, loc=0, scale=1)
        q = weibull_min.fit(dist, loc=0, scale=1)
        r = genextreme.fit(dist)
        
        fig, ax = plt.subplots()
        if cutOffSpeed is not None:
            ax1 = ax.twinx()
            ax1.get_yaxis().set_visible(False)
            ax1.plot((cutOffSpeed,cutOffSpeed), (0.1, 1000), '-', lw= 1, color='black', alpha = 0.5, label='cutoff')
                
        ax.hist(dist, bins=int(dist.max() - dist.min()+1), alpha = 0.85, label = 'data', density = True)
        
        X = np.linspace(0.01, int(dist.max())+1, int(dist.max())*4)
        ax.plot(X, gumbel_r.pdf(X, *n), 'r-', lw=2, alpha=0.8, label='gumbel')
        ax.plot(X, exponweib.pdf(X, *p), 'b-', lw=2, alpha=0.8, label='exponweibull')
        ax.plot(X, weibull_min.pdf(X, *q), 'g-', lw=2, alpha=0.8, label='weibull')
        ax.plot(X, genextreme.pdf(X, *r), '-', color = 'grey', lw=2, alpha=0.8, label='genextreme')

        ax.set_yscale(scale)
        
        title = ''
        if stnNo is not None: title += f"stn - {stnNo}"
        if data is not None: title += f" ({data})"
        if title != '': ax.set_title(title)
        
        fig.legend()
        plt.show()
    
    if cutOffSpeed is not None:
        return 1 - gumbel_r.cdf(cutOffSpeed, *n)
    elif percentile is not None: 
        return gumbel_r.ppf(percentile, *n)
        # return genextreme.ppf(percentile, genextreme.fit(dist)[0])
    return None
    
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
    
    stn = readAll(r'BOM Wind Data', speedThreshold, multiprocess=True, goodnessOfFit=genextreme)
    stn.to_csv(r'Results/statModels/genextreme.csv')
    stn = readAll(r'BOM Wind Data', speedThreshold, multiprocess=False, goodnessOfFit=exponweibull)
    stn.to_csv(r'Results/statModels/exponweibull.csv')
    stn = readAll(r'BOM Wind Data', speedThreshold, multiprocess=False, goodnessOfFit=weibull_min)
    stn.to_csv(r'Results/statModels/weibullmin.csv')    
    stn = readAll(r'BOM Wind Data', speedThreshold, multiprocess=False, goodnessOfFit=weibull_max)
    stn.to_csv(r'Results/statModels/weibullmax.csv')
    
    # stn = readAll(r'BOM Wind Data', speedThreshold, multiprocess=True)
    # stn = readData(r'BOM Wind Data\AWS_Wind-NT', speedThreshold, multiprocess=True)   

    # stn.to_csv(r'Data1/WindStats.csv', index = False)
    
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
