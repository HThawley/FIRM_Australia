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
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon


#%%

def readAll(location, speedThreshold, multiprocess=False):
    active_dir = os.getcwd()
    os.chdir(location)
    
    folders = [(path, speedThreshold, True) for path in os.listdir() if ('.zip' not in path) and ('MaxWindGust' not in path)]
    
    if multiprocess:
        print('Reading and processing data\nCompleted: ', end = '')
        with NestablePool(max_workers=min(cpu_count(), len(folders))) as processPool:
            results = processPool.map(readData, folders)
        print('')
    else: 
        results = []
        for folderTuple in tqdm(folders, desc = 'Reading wind data folder-by-folder:'):
            results.append(readData(folderTuple))

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

def get_meanSpeed100m(stn):
    # The multiprocessing library doesn't play nicely with os.chdir(), hence a full path 
    try:
        windMap = rasterio.open('/media/fileshare/FIRM_Australia_Resilience/Data/AUS_wind-speed_100m.tif')
    except FileNotFoundError:
        windMap = rasterio.open(r'C:\Users\hmtha\OneDrive\Desktop\FIRM_Australia\Data\AUS_wind-speed_100m.tif')

    coord_list = [(x,y) for x, y in zip(stn['longitude'], stn['latitude'])]

    stn['meanSpeed-100m'] = [x[0] for x in windMap.sample(coord_list)]
    stn['meanSpeed-100m'] = pd.to_numeric(stn['meanSpeed-100m'])

    windMap.close()   
    return stn

def readData(argTuple, folder=None, speedThreshold=None, multiprocess=True):
    if argTuple: 
        folder, speedThreshold, multiprocess = argTuple
    active_dir = os.getcwd()
    os.chdir(folder)
    
    stn = readStnDetail(folder)
    stn = get_meanSpeed100m(stn)
    
    argTuples = [(path, stn, speedThreshold) for path in os.listdir() if 'Data' in path]

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
        with Pool(processes = min(cpu_count(), len(argTuples))) as processPool:
            for inst in processPool.imap_unordered(readFile, argTuples):
                result.append(inst)
    else: 
        for argTuple in argTuples:
            result.append(readFile(argTuple))
            
    result = pd.DataFrame(result, 
                          columns = ['station no.', 'meanDuration', 'highWindFrac', 'scaleFactor',
                                     'meanSpeed-10m', 'startTime', 'meanRes', 'Observations'])
    stn = stn.merge(result, on = 'station no.', how = 'outer', indicator ='indicator')
    
    assert len(stn['indicator'].unique()) == 1 
    assert stn['indicator'].unique()[0] == 'both'
    
    stn = stn.drop(columns = 'indicator')
    
    print(f'{folder[9:]}; ', end='')
    os.chdir(active_dir)
    return stn    
    
def readFile(argTuple):
        
    path, stn, speedThreshold = argTuple
    stnNo = path[11:17]
    
    x = pd.read_csv(path, usecols = [2,3,4,5,6,12,16], dtype = str)
    
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors = 'coerce')
        
    x = x.rename(columns={'Speed of maximum windgust in last 10 minutes in  km/h':'gustSpeed-10m', 
                          'Wind speed in km/h':'meanSpeed-10m'})
    
    x['dt'] = x[x.columns[0:5]].apply(lambda row: lambdaDt(*row), axis=1)
    
    # x['speed'] = pd.to_numeric(x['speed'], errors = 'coerce')
    x['gustSpeed-10m'] = x['gustSpeed-10m'] / 3.6# km/h to m/s
    x['meanSpeed-10m'] = x['meanSpeed-10m'] / 3.6# km/h to m/s
    
    x = x[['dt', 'gustSpeed-10m', 'meanSpeed-10m']]

    if len(x.dropna()) == 0: 
        return [stnNo, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]
    
    longTermMeanSpeed = x['meanSpeed-10m'].dropna().mean()
    startTime = x.dropna()['dt'].min()
    meanRes = (x.dropna()[['dt']].diff(periods=1, axis=0).sum()[0]/len(x)).total_seconds()/60
    
    try: 
        scaleFactor = stn[stn['station no.']==stnNo]['meanSpeed-100m'].values[0] / longTermMeanSpeed
    except KeyError: 
        return [stnNo, pd.NA, pd.NA, pd.NA, longTermMeanSpeed, startTime, meanRes, len(x)]
    if pd.isna(scaleFactor): 
        return [stnNo, pd.NA, pd.NA, pd.NA, longTermMeanSpeed, startTime, meanRes, len(x)]
    
    x['gustSpeed-100m'] = scaleFactor * x['gustSpeed-10m']
    
    highWindIntegral = gumbelModelling(x['gustSpeed-100m'].dropna(), speedThreshold[0])
    
    #define storm indicator initially
    x['highSpeedInd'] = x['gustSpeed-100m'].apply(lambda gust: 2 if gust >=speedThreshold[0] else 1 if gust>=speedThreshold[1] else 0)
    #smooth storm indicator to include adjacent instances within 10% of cutOffSpeed
    x['highSpeedInd'] = x['highSpeedInd'].rolling(window=2).mean().apply(lambda ind: 1 if ind>=1.5 else 0)
    
    #https://stackoverflow.com/questions/37934399/identifying-consecutive-occurrences-of-a-value-in-a-column-of-a-pandas-dataframe
    x['duration'] = x['highSpeedInd'].groupby((x['highSpeedInd'] != x['highSpeedInd'].shift()).cumsum()).transform('size') * x['highSpeedInd']
    
    lengths = x['duration'].value_counts().drop(index=0)
    meanDuration = np.mean(lengths/lengths.index)
    
    return [stnNo, meanDuration, highWindIntegral, scaleFactor, longTermMeanSpeed, startTime, meanRes, len(x)]

def lambdaDt(y, mo, d, h, mi): return dt(y, mo, d, h, mi)

def gumbelModelling(dist, cutOffSpeed, plot=False):    
    a, c = gumbel_r.fit(dist)
        
    if plot:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.hist(dist, bins=50, alpha = 0.5)
        x = np.linspace(gumbel_r.ppf(0.000001, a, c), gumbel_r.ppf(0.999999, a, c), 200)
        ax1.plot(x, gumbel_r.pdf(x, a, c), 'r-', lw=2, alpha=0.9, label='gumbel pdf')
    
    return 1 - gumbel_r.cdf(cutOffSpeed, a, c)


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
    
    stnCopy['distance'] = stnCopy[['latitude', 'longitude']].apply(lambda x: Haversine(*x, lat, lon), axis = 1)
    
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
    speedThreshold=(25, #turbine cut-off speed 25 m/s
                    25*0.9 #wind gust speed tolerance, 10% 
                    )
    
    stn = readAll(r'BOM Wind Data', speedThreshold, multiprocess=True)
    # stn = readData(r'BOM Wind Data\AWS_Wind-NT', speedThreshold, multiprocess=True)   

    stn.to_csv(r'Data/WindStats.csv', index = False)
#%%
def runGeoAnalysis(stn):
    geoMap = gpd.read_file(r'Geometries/australia.geojson')
       
    stn = pd.read_csv(r'Data/WindStats.csv')
    
    stn = stn.dropna()
    stn['mainland'] = stn[['longitude', 'latitude']].apply(lambda coord: geoMap.contains(Point(coord['longitude'], coord['latitude'])), axis=1)
    stn = stn[stn['mainland']]
    
    return stn, geoMap

#%%

def plotMap(stn, geoMap):
    active_dir = os.getcwd()
    os.chdir('Geometries/wind')
    
    fig, ax = plt.subplots(figsize=(15,15), dpi = 1500)
    ax.grid(alpha = 0.6, color = 'black', linewidth = 1)
    
    geoMap.plot(ax = ax)
    
    ax.scatter(x = stn['longitude'], y = stn['latitude'], color = 'red', alpha = 0.5, s =15)
    
    for zone in os.listdir():
        zone = gpd.read_file(zone)
        zone.plot(ax = ax, color = 'green', alpha = 0.3)
    
    ax.set_xticks(range(110, 155, 5))
    ax.set_yticks(range(-45, -5, 5))
    plt.show()
    
    os.chdir(active_dir)


