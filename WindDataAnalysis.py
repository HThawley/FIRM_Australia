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
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay
import rasterio
import geopandas as gpd
from shapely.geometry import Point, Polygon#, MultiPolygon
from shapely.ops import nearest_points
from shapely import distance
import warnings



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

def formatStnNo(col):
    outCol = col.apply(lambda x: '0'*(6-len(str(int(x))))+str(int(x)) 
                    if not pd.isna(x)
                    else pd.NA)
    return outCol

def readStnDetail(folder):
    
    stnDet = [path for path in os.listdir() if 'StnDet' in path]
    
    assert len(stnDet) == 1
    
    stn = pd.read_csv(stnDet[0],
                      header = None,
                      usecols = [1,3,6,7,10])
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
                                     'meanSpeed-10m', 'startTime', 'meanRes', 'Observations', 'len(lengths)', 'lengths'])
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
    startTime = x.dropna(subset=['gustSpeed-10m'])['dt'].min()
    meanRes = (x.dropna(subset=['gustSpeed-10m'])[['dt']].diff(periods=1, axis=0).sum()[0]/len(x)).total_seconds()/60
    
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
    if pd.isna(meanDuration):
        meanDuration=0
    
    return [stnNo, meanDuration, highWindIntegral, scaleFactor, longTermMeanSpeed, startTime, meanRes, len(x.dropna()), len(lengths), lengths]

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

def hav(angle): return np.sin(angle/2)**2

def Haversine(lat, lon, lat2, lon2):
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
    lat, lon, lat2, lon2 = map(np.radians, [lat, lon, lat2, lon2])

    a = hav(lat - lat2) + np.cos(lat) * np.cos(lat2) * hav(lon - lon2)
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = 6371.0088 * c
    return d

def derivativeHaversineLatitude(lat, lon, lat2, lon2):
    """
    wolfram alpha gives this as the derivative (w.r.t. a) where a=lat, b=lat2, c=lon1, d=lon2
    
    2 (sin(a - b) - 2 sin(a) cos(b) hav(c - d))/(4 sqrt(-(cos(a) cos(b) hav(c - d) + hav(a - b) - 1) (cos(a) cos(b) hav(c - d) + hav(a - b))))
    """
    return (6371.0088 * 2 * np.divide(
            
        np.sin(lat-lat2) - 2 * np.sin(lat) * np.cos(lat2) * hav(lon-lon2),
            
        4*((-(np.cos(lat) * np.cos(lat2) * hav(lon-lon2) + hav(lat-lat2) - 1) *(
              np.cos(lat) * np.cos(lat2) * hav(lon-lon2) + hav(lat-lat2)))**0.5)))

def derivativeHaversineLongitude(lat, lon, lat2, lon2):
    """
    wolfram alpha gives this as the derivative (w.r.t. c) where a=lat1, b=lat2, c=lon, d=lon2
 
    2 (cos(a) cos(b) sin(c - d))/(4 sqrt(-(cos(a) cos(b) hav(c - d) + hav(a - b) - 1) (cos(a) cos(b) hav(c - d) + hav(a - b))))
    """

    return (6371.0088 * 2 * np.divide(
        
        np.cos(lat) * np.cos(lat2) * np.sin(lon-lon2),
        
        4*((-(np.cos(lat) * np.cos(lat2) * hav(lon-lon2) + hav(lat-lat2) - 1) *(
              np.cos(lat) * np.cos(lat2) * hav(lon-lon2) + hav(lat-lat2)))**0.5)))


    
#%%
def filterBadStations(stn):
    geoMap = gpd.read_file(r'Geometries/australia.geojson')
    stn = stn.dropna()
    
    warnings.filterwarnings('ignore', category = pd.errors.SettingWithCopyWarning)
    stn['mainland'] = stn[['longitude', 'latitude']].apply(lambda coord: geoMap.contains(Point(*coord)), axis=1)
    warnings.filterwarnings('default', category = pd.errors.SettingWithCopyWarning)

    stn = stn[stn['mainland']]
    stn = stn.drop(columns = ['mainland'])
    return stn

def lambdaDistanceEdge(point, polygon): 
    p1, p2 = nearest_points(polygon, point)
    #x is longitude, y is latitude
    return Haversine(p1.y, p1.x, p2.y, p2.x)

def lambdaDistancePoints(p1, p2):
    return Haversine(p1.x, p1.y, p2.x, p2.y)

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

def plotMap(stn):
    # global fig, ax 
    geoMap = gpd.read_file(r'Geometries/australia.geojson')
    active_dir = os.getcwd()
    
    os.chdir('Geometries/wind')
    
    fig, ax = plt.subplots(figsize=(15,15), dpi = 1500)
    ax.grid(alpha = 0.5, color = 'black', linewidth = 1)
    
    geoMap.plot(ax = ax)
    
    ax.scatter(x = stn['longitude'], y = stn['latitude'], color = 'red', alpha = 0.9, s =4) 
    
    warnings.filterwarnings('ignore', category = UserWarning)
    for zone in os.listdir():
        poly = gpd.read_file(zone)     
        poly.plot(ax = ax, color = 'green', alpha = 0.3)
        ax.scatter(poly.centroid.x, poly.centroid.y, color = 'black', alpha=1, s=5)
    warnings.filterwarnings('default', category = UserWarning)

    ax.set_xticks(np.divide(range(110*2, 160*2, 5),2.))
    ax.set_yticks(np.divide(range(-45*2, -5*2, 5),2.))
    plt.show()
    
    os.chdir(active_dir)


def interpolate(stn, poly):
    coords = np.array(list(zip(stn['longitude'], stn['latitude'])))
    highWindFrac = np.array(stn['highWindFrac'])
    meanDuration = np.array(stn['meanDuration'])
    centroid = poly.centroid.x, poly.centroid.y
    
    if coords.shape[0] > 3: 
        hull = ConvexHull(coords)
        highWindFrac = LinearNDInterpolator(coords, highWindFrac)
        meanDuration = LinearNDInterpolator(coords, meanDuration)

        if Delaunay(hull.points).find_simplex(centroid) >= 0:
            #centroid within interpolation space
            highWindFrac = highWindFrac(*centroid)
            meanDuration = meanDuration(*centroid)
        
        else: #centroid not within interpolation space
            #find nearest point inside interpolation space
            nearNeighbour = nearest_points(poly.centroid, Polygon(hull.points[hull.vertices]))[1]
            #Take the nearest point (nearest neighbour), rounding avoids issue where point 
            # is ~10^-15 m away from the interpolating area
            i = 20
            while i > 2:    
                hwf = highWindFrac(round(nearNeighbour.x,i), round(nearNeighbour.y,i))
                mdr = meanDuration(round(nearNeighbour.x,i), round(nearNeighbour.y,i))
                if pd.isna(hwf) or pd.isna(mdr):
                    i-=1
                    continue
                else: 
                    highWindFrac, meanDuration = hwf, mdr
                    break
    else: 
        highWindFrac = distanceWeightedAverage(coords, highWindFrac, centroid)
        meanDuration = distanceWeightedAverage(coords, meanDuration, centroid)
    
    zoneDf = pd.DataFrame(
        [[stn['closestZone'].unique()[0], 
         highWindFrac, 
         meanDuration]], 
        columns = ['zone', 'highWindFrac', 'meanDuration'])
    
    return zoneDf

def distanceWeightedAverage(points, values, sample_point):    
    weights = 1/ Haversine(points[:,0], points[:,1], sample_point[0], sample_point[1])
    
    return (sum(values * weights) / sum(weights))    




def zoneAnalysis(stn):
    active_dir = os.getcwd()
    os.chdir('Geometries/wind')

    zoneDf = pd.DataFrame([])
        
    for zone in os.listdir():
        poly = gpd.read_file(zone)['geometry'][0]
        zone = int(zone.split('.')[0])
        
        zoneDf = pd.concat([zoneDf, interpolate(stn[stn['closestZone'] == int(zone)], poly)])
    
    os.chdir(active_dir)
    return zoneDf

def removeAnomalousStns(stn):
    """removes stations which are very different to nearby stations."""
    
    badStns = ['053000',#only 16068 observations
               '041560',#only 26776 observations
               '068076', #only 2746 observations
               '030024', #only 24 observations
               '016092', #only 33740 observations, and results not very well in line with nearby stations
               '023849', #only 208 observations and not well in line with nearby stations 023849
               '018207', #only 32425 observations and not well in line with nearby 018200 and 018012
               '092037', #only 36 observations 
               #'097085', #although this is geographically very different to nearby sites, 
               #geography is closer to that of wind farms and results are not very anomalous
               '094250',#11972 obs
               '091375',#5587 obs
               '092133',#24694 obs
               '092163',#12619 obs
               #'094087,#although geograpgicall very different to nearby sites, geography is closer
               #to that of wind farms and results are rather similar to (e.g.) 094195, 094008, 092100, 096003
               '087185',#22956 obs
               '078072',#2648 obs
               '078031'#42 obs
               ]
    
    stn = stn[~stn['station no.'].isin(badStns)]
    return stn

#%%
 
if __name__=='__main__':

    speedThreshold=(25, #turbine cut-off speed 25 m/s
                    25*0.9 #wind gust speed tolerance, 10% 
                    )
    
    stn = readAll(r'BOM Wind Data', speedThreshold, multiprocess=True)
    # stn = readData(r'BOM Wind Data\AWS_Wind-NT', speedThreshold, multiprocess=True)   

    stn.to_csv(r'Data/WindStats.csv', index = False)
    stn = pd.read_csv(r'Data/WindStats.csv')
    stn['station no.'] = formatStnNo(stn['station no.'])
    # stn = filterBadStations(stn)
    stn = findClosestZones(stn, 50) #km
    stn = stn.dropna(subset=['highWindFrac', 'meanDuration'], how='any')

    stn = removeAnomalousStns(stn)

    zones = zoneAnalysis(stn).sort_values('zone').reset_index(drop=True) 
    zones.to_csv('Results/windDataByZone/_zoneData_.csv', index=False)
    
    # plotMap(stn)     

    #Manual Analysis
    # grpby = stn.groupby('closestZone')[['meanSpeed-10m','meanSpeed-100m','meanDuration',
    #     'highWindFrac','scaleFactor','meanRes','Observations']].describe()
    # grpby = grpby.drop(columns=[col for col in grpby.columns if '%' in col[1] \
    #         or 'std' in col[1] or 'count'==col[1] and 'meanSpeed-10m'!=col[0]])
    # grpby.to_csv('Results/zoneWindStats.csv')
    
    # for i, df in stn.groupby('closestZone'):
    #     df.to_csv(f'Results/windDataByZone/Zone{i}.csv', index=False)
