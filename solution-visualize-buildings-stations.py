# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 15:07:49 2016

@author: Mike
"""
import geocoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import inspect
from geopy.distance import vincenty
from mpl_toolkits.basemap import Basemap
import weather
from test_weather import decompose_daily, getCorrCoeff


def main():
    # locate the meta data file
    # get the current file location
    scriptName = inspect.getfile(inspect.currentframe())
    scriptDir = os.path.abspath(os.path.join(scriptName, os.pardir))
    parentDir = os.path.dirname(scriptDir)
    
    # locate the meta data
    fname = 'all_sites.csv'
    metaDataPath = os.path.join(parentDir, 'csv-only',
                                'meta', fname)
    
    
    # locate tmy3 meta dat
    fname = 'TMY3_StationsMeta.csv'
    tmy3DataPath = os.path.join(parentDir, 'csv-only',
                                'meta', fname)
    
	# Read the data into a pandas dataframe
	tmy3MetaData = pd.DataFrame.from_csv(tmy3DataPath,index_col=None)   
    metaData = pd.DataFrame.from_csv(metaDataPath,index_col=None)
    
    # get location data
    lat = metaData[u'LAT'].values
    lng = metaData[u'LNG'].values

    # Find closest TMY3 weather data to each building
    min_distances = []
    tmy3_file = []
    tmy3_lat = tmy3MetaData['Latitude'].values
    tmy3_lng = tmy3MetaData['Longitude'].values
	
	# for each pair of coordinates
    for pair in zip(lat, lng):
        dis = []
		# calculate the distance to each weather station
        for tmy3_loc in list(zip(tmy3_lat, tmy3_lng)):
            dis.append(vincenty(pair, tmy3_loc).miles)
		
		# find the minimum distance and the index
        dis = np.array(dis)
        min_loc = (np.min(dis), np.argmin(dis))
        
		# store results
        min_distances.append(min_loc)
        tmy3_file.append([tmy3MetaData['Site Name'][min_loc[1]],
                          tmy3MetaData['State'][min_loc[1]],
                          tmy3MetaData['USAF'][min_loc[1]]])
    
	# get unique airport data
    airport_names = set([x[0] for x in tmy3_file])
	
	# get a boolean vector of True/False for each row in the 
	# original dataframe.
    criterion = tmy3MetaData['Site Name'].map(lambda x: x in airport_names)
	
	# subset the original dataframe
    unique_tmy3 = tmy3MetaData[criterion]
	
	# get the relevant latitudes and longitudes
    uniTmyLng = unique_tmy3['Longitude'].values
    uniTmyLat = unique_tmy3['Latitude'].values        
    
    # plot a map of the united states
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    my_map = Basemap(projection = 'merc', lat_0=39.8, lon_0=98.6,
                     resolution='l', area_thresh=1000, llcrnrlon=-125, llcrnrlat=20,
                     urcrnrlon=-60, urcrnrlat = 50, ax=ax)
    my_map.drawcoastlines()
    my_map.drawcountries()
    my_map.drawstates()
    my_map.fillcontinents(color='gray', alpha=0.65)
    my_map.drawmapboundary()
    my_map.drawmeridians(np.arange(0, 360, 30))
    my_map.drawparallels(np.arange(-90, 90, 30))
    x,y = my_map(lng,lat)
    my_map.plot(x, y, 'bo', markersize=8, label='EnerNOC Building')
    x,y = my_map(uniTmyLng, uniTmyLat)
    my_map.plot(x,y,'yo', markersize=5, label='TMY3 Station')
    plt.legend(frameon=False, loc=0)
    plt.savefig('../figures/buildingslocs.png')
    

if __name__ == '__main__':
    main()