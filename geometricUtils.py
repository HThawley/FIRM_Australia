# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:56:27 2023

@author: hmtha
"""

import numpy as np

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

def distanceWeightedAverage(points, values, sample_point):    
    weights = 1/ Haversine(points[:,0], points[:,1], sample_point[0], sample_point[1])
    
    return (sum(values * weights) / sum(weights))
