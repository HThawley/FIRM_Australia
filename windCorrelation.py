# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:11:48 2023

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

def readAll(location, stn, multiprocess=False):
    active_dir = os.getcwd()
    os.chdir(location)
    
    folders = [(path, stn, False) 
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
    
    meanSpeeds = pd.concat(results, axis = 1)
    
    os.chdir(active_dir)
    return meanSpeeds

def stormSpeed(dist, perc): 
    return dist.tail(int(perc*len(dist))).iloc[0]

def formatStnNo(col):
    return col.apply(lambda x: '0'*(6-len(str(int(x))))+str(int(x)) if not pd.isna(x) else pd.NA)

def readData(argTuple, folder=None, stn=None, multiprocess=None):
    if argTuple is not None: 
        folder, stn, multiprocess = argTuple
    for arg in (folder, stn, multiprocess):
        assert arg is not None
    active_dir = os.getcwd()
    os.chdir(folder)
    
    argTuples = [(path, stn) for path in os.listdir() if 'Data' in path]
    
    if multiprocess:
        with Pool(processes = min(cpu_count(), len(argTuples))) as processPool:
            resultGen = processPool.map(readFile, argTuples)
            results = list(resultGen)

    else: 
        results = [readFile(argTuple) for argTuple in argTuples]
    
    meanSpeeds = pd.concat(results, axis = 1)

    os.chdir(active_dir)
    return meanSpeeds

    
def readFile(argTuple=None, path=None, stn=None):
    if argTuple is not None:
        path, stn = argTuple
    for arg in (path, stn):
        assert arg is not None

    stnNo = path[11:17]
    if stnNo not in stn['station no.'].unique():
        return pd.DataFrame({})
    
    df = pd.read_csv(path, usecols = [7,8,9,10,11,12,16], dtype = str)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')  
    df = df.rename(columns={'Wind speed in km/h':stnNo})
    
    df['dt'] = df[df.columns[0:5]].apply(lambda row: lambdaDt(*row), axis=1)
    
    df[stnNo] = df[stnNo] / 3.6# km/h to m/s
    df = df.dropna(subset=stnNo)

    if len(df.dropna()) == 0: 
        return pd.DataFrame([])
    
    df = df.sort_values(by=stnNo).drop_duplicates(subset='dt', keep = 'last').sort_values(by = 'dt')
    df.index = df['dt']
    return df[[stnNo]]

def lambdaDt(y, mo, d, h, mi): return dt(y, mo, d, h, mi)

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
            
    os.chdir(active_dir)
    return stn

def findClosestZones(stn, distanceThreshold):

    stn['point'] = stn[['longitude', 'latitude']].apply(lambda coord: Point(*coord), axis = 1)
    
    stn['closestZone'], stn['distanceToZone'], stn['distanceToCentroid'] = pd.NA, np.inf, np.inf
    
    active_dir = os.getcwd()
    os.chdir('Geometries/wind')
    
    for zone in os.listdir():
        poly = gpd.read_file(zone)['geometry'][0]
        distances = stn['point'].apply(lambda point: lambdaDistanceEdge(point, poly))
        centroidDistance = stn['point'].apply(lambda point: lambdaDistancePoints(point, poly.centroid))
        inZone = stn['point'].apply(lambda point: poly.contains(point))
        
        distanceMask = (distances < stn['distanceToZone'])
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
    
def removeAnomalousStns(stn):
    badStns = ['053000','041560','068076','030024','016092','023849','018207', 
               '092037','094250','091375','092133','092163','087185','078072','078031']
    return stn[~stn['station no.'].isin(badStns)]

#%%

if __name__ == '__main__':
    stn = readStnDetail(r'BOM Wind Data')
    stn = findClosestZones(stn, 50)
    stn = removeAnomalousStns(stn)
    
    meanSpeeds = readAll(r'BOM Wind Data', stn, multiprocess=True)
#%%

    
    for col in meanSpeeds.columns:
        if col not in stn['station no.'].unique():
            meanSpeeds = meanSpeeds.drop(columns=col)
        
#%%
    corr = meanSpeeds.corr()
    for i in range(len(corr)):
        corr.iloc[i,i] = 0 
    
    corrThresh = 0.75
    corr = pd.DataFrame([(col, list(corr.index[corr[col]>corrThresh])) for col in corr.columns], 
                        columns = ['station no.', f'corrThresh - {corrThresh}'])
    
    corr = corr.explode(f'corrThresh - {corrThresh}', ignore_index=True).dropna()
    
    corr = pd.merge(stn, corr, on='station no.', how='outer')
    corr = pd.merge(corr[['station no.', 'closestZone']], corr, left_on='station no.', right_on= f'corrThresh - {corrThresh}', how = 'inner')
    
    corr = corr[corr['closestZone_x'] != corr['closestZone_y']]
    corr['pair'] = corr[['closestZone_x', 'closestZone_y']].apply(lambda row: str(list(row.sort_values())),axis = 1)
    
    identicalZones = ('[3, 27]', '[6, 17]')
    corr = corr[~corr['pair'].isin(identicalZones)]
    
    corr = pd.concat([corr, corr['pair'].str.replace('[','').str.replace(']','').str.split(',', expand = True)], axis = 1)
    corr = corr.rename(columns={0:'zone2', 1:'zone1'})
    
    corr['zone1'], corr['zone2'] = corr['zone1'].astype(int), corr['zone2'].astype(int)
     
    zd = pd.read_csv(r'C:/Users/hmtha/Desktop/ENGN4712/Zone Dict.csv', usecols=[8,9])
    zd.columns = ['zoneNo', 'name']   
    
    corr = pd.merge(corr, zd, left_on = 'zone1', right_on='zoneNo', suffixes=('', '1')).drop(columns=['zoneNo'])
    corr = pd.merge(corr, zd, left_on = 'zone2', right_on='zoneNo', suffixes=('', '2')).drop(columns=['zoneNo'])
    
    
    
