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
    # Color according to industry.  
    #---------------------------------------------------------------------------
    # in this section you need to determine how many unique industries are
    # represented and to then give each one a unique color.
    # It may be useful to keep this information in a dictionary that relates
    # the industry type to the corresponding color.
    # Search for documentation on color maps in pyplot for more details and
    # examples.    
        
    
    # How many unique industries are there?
    
    # for each industry get a color from a diverging color map
    
    # relational dictionary
    
    #---------------------------------------------------------------------------
    # Size According to energy use intensity
    #---------------------------------------------------------------------------
    '''
    Next it's time to determine the energy use intensity of each building and
    plot the size of each point based on this value.  The data for each building
    is stored in '../csv-only/csv/[SITE_ID].csv' and the 'values' column is in
    kWh.
    
    The energy use intensity is the sum of the energy use at each interval divided
    by the square footage of the building.  How can we get each of these pieces
    of information?
    '''
    
    # for each building
        # get the site_id
    
        # get the square footage
    
        # define the file path
    
        # read in the energy data
    
        # calculate the EUI
    
        # store the EUI

    
    # the EUI data has to be scaled in order to show up on the plot.
    # Create a scaled data set to your liking and see this post on stack overflow
    # http://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
   
  
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
    
    # convert lat, lng to plot coordinates

    # for each building
        # plot to my_map.  Color the marker based on the 'INDUSTRY' and 
        # give marker size based on the EUI calculated for that building.    
        
    # Add weather station locations to the plot
    x,y = my_map(uniTmyLng, uniTmyLat)
    my_map.plot(x,y,'yo', markersize=5, label = 'TMY3 Weather Station')
    
    #---------------------------------------------------------------------------    
    # create legend handles and labels
    #---------------------------------------------------------------------------
    '''
    You most likely will be having legend issues if you labeled each building
    as you plotted.  Now is where we fix this (or avoid adding labels prior to
    this section).
    '''    
    # get the existing handles
    # see matplotlib.org/users/legend_guide.html for more details
    
    # create custom handles for the building types
    # see matplotlib.org/users/legend_guide.html for more details
    
    
    # Turn on the legend in best location, adjust legend display for a single point
    # like the final plot
    # see matplotlib.org/users/legend_guide.html for more details on HandlerLine2D class

    
    # Ensure the required directory exists
    if not os.path.isdir('../../figures'):
		os.mkdir('../../figures')
    
    # Save figure
    plt.savefig('../../figures/buildingslocs-session2.png')
    

if __name__ == '__main__':
    main()