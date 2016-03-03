# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 15:07:49 2016

@author: Mike
"""

import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import os
from geopy.distance import vincenty
from mpl_toolkits.basemap import Basemap

def findClosestStation(coords1, coords2):
    '''
    Function finds the closest value of coords2 to each value in coords1.
    Inputs:
    coords1 (iterable) - contains (lat, long) pairs
    coords2 (iterable) - contains (lat, long) pairs
    
    Outputs:
    closest (list) - contains tuple of index and value of coords2 that 
    corresponds to the closest (lat, long) pair for each index in coords1
    '''
    
    closest = []
	# for each pair of coordinates
    for firLoc in coords1:
        dis = []
        # calculate the distance to each weather station
        for secLoc in coords2:
            dis.append(vincenty(firLoc,secLoc).miles)
		
        # find the minimum distance and the index
        # Uses base python, but numpy.argmin is applicable
        # Check documentation on built-in functions for min and enumerate
        min_index, min_distance = min(enumerate(dis), key = lambda p: p[1])
        
        # store results
        closest.append((min_index, min_distance))
    
    return closest

def main():
   
    # locate the meta data
    fname = 'all_sites.csv'
    metaDataPath = os.path.join(os.pardir, os.pardir, 'csv-only',
                                'meta', fname)
    
    
    # locate tmy3 meta dat
    fname = 'TMY3_StationsMeta.csv'
    tmy3DataPath = os.path.join(os.pardir, os.pardir,'csv-only',
                                'meta', fname)
    
    # Read the data into a pandas dataframe
    tmy3MetaData = pd.DataFrame.from_csv(tmy3DataPath, index_col=None)   
    metaData = pd.DataFrame.from_csv(metaDataPath, index_col=None)
    
    # get location data
    lat = metaData[u'LAT'].values
    lng = metaData[u'LNG'].values

    # Find closest TMY3 weather data to each building
    tmy3_lat = tmy3MetaData['Latitude'].values
    tmy3_lng = tmy3MetaData['Longitude'].values
	
    min_distance = findClosestStation(list(zip(lat,lng)),
                                      list(zip(tmy3_lat,tmy3_lng)))
    
    # store unique attributes of each minimum distance station
    tmy3SiteNames = [tmy3MetaData['Site Name'][x[0]] for x in min_distance]   
    
    # get unique airport data
    airportUniNames = set(tmy3SiteNames)
    
    # get a boolean vector of True/False for each row in the 
    # original dataframe.
    
    # Review the map method of pandas.DataFrame objects for any questions
    criterion = tmy3MetaData['Site Name'].map(lambda x: x in airportUniNames)
	
    # subset the original dataframe
    unique_tmy3 = tmy3MetaData[criterion]
	
    # get the relevant latitudes and longitudes
    uniTmyLng = unique_tmy3['Longitude'].values
    uniTmyLat = unique_tmy3['Latitude'].values

    #---------------------------------------------------------------------------
    # Color according to industry.  Size According to energy use intensity
    #---------------------------------------------------------------------------
    
    # How many unique industries are there?
    industryNames = set(metaData[u'INDUSTRY'].values)
    numIndustries = len(industryNames)
    
    # for each industry get a color from a diverging color map
    cm = plt.get_cmap('Set1')
    colPts = np.linspace(0.0, 0.5, numIndustries)
    
    # relational database
    type_color_map = dict(zip(industryNames, cm(colPts)))
    

    
    # point sizes according to energy use intensity
    buiEUI = []
    for index in range(len(lng)):
        site_id = metaData[u'SITE_ID'][index]
        flrArea = metaData[u'SQ_FT'][index]
        fpath = os.path.join(os.pardir, os.pardir, 'csv-only', 'csv',
                             '{}.csv'.format(site_id))
        bui_energy_kWh = np.genfromtxt(fpath, delimiter=',', skip_header=1, 
                                       usecols = 2)
        bui_energy_kBtu = np.sum(bui_energy_kWh*3.412)
        buiEUI.append(bui_energy_kBtu/flrArea)
    
    buiEUI_scaled = [np.interp(kk, [min(buiEUI), np.percentile(buiEUI, 25), 
                                    np.percentile(buiEUI, 75),
                                    max(buiEUI)],
                                    [5, 10, 20, 40]) for kk in buiEUI]    
  
    #---------------------------------------------------------------------------
    # Final plotting of the map
    #---------------------------------------------------------------------------    
    # set up plotting scene    
    fig = plt.figure(figsize=(11,8.5))
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    
    # plot a map of the united states
    my_map = Basemap(projection = 'merc', lat_0=39.8, lon_0=98.6,
                     resolution='l', area_thresh=1000, llcrnrlon=-125,
                     llcrnrlat=20, urcrnrlon=-60, urcrnrlat = 50, ax=ax)
    
    # Add clarifying identifiers to map
    my_map.drawcoastlines()
    my_map.drawcountries()
    my_map.drawstates()
    my_map.fillcontinents(color='gray', alpha=0.0)
    my_map.drawmapboundary()
    my_map.drawmeridians(np.arange(0, 360, 30))
    my_map.drawparallels(np.arange(-90, 90, 30))
    
    # Add building locations to the plot   
    x,y = my_map(lng,lat)
    for index in range(len(lng)):
        building_type = metaData[u'INDUSTRY'][index]
        my_map.plot(x[index], y[index], ls = 'None',alpha=0.5, 
                    marker = 'o', markerfacecolor = type_color_map[building_type],
                    markeredgecolor = type_color_map[building_type],
                    markersize=buiEUI_scaled[index])
    
    # Add weather station locations to the plot
#    x,y = my_map(uniTmyLng, uniTmyLat)
#    my_map.plot(x,y,'yo', markersize=5, label = 'TMY3 Weather Station')
    
    # create legend handles and labels
    # get the existing handles
    handles, labels = ax.get_legend_handles_labels()
    
    # create custom handles for the building types
    for key in type_color_map:
        h = mlines.Line2D([],[], markersize = 5.0,
                          marker = 'o', ls='None', markerfacecolor = type_color_map[key],
                          markeredgecolor = type_color_map[key],
                          label=key)
        handles.append(h)
        
    # Turn on the legend in best location, adjust legend display
    # see matplotlib.org/users/legend_guide.html for more details on HandlerLine2D class
    plt.legend(handles=handles, frameon=False, loc=0,
               handler_map = {type(handles[0]):HandlerLine2D(numpoints = 1)})
    
    # Ensure the required directory exists
    if not os.path.isdir('../../figures'):
		os.mkdir('../../figures')
    
    # Save figure
    plt.savefig('../../figures/buildingslocs-session2.png')
    

if __name__ == '__main__':
    main()