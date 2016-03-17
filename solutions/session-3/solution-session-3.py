# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 15:07:49 2016

@author: Mike
"""
from __future__ import division
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
    closest (list) - contains tuple of (index, distance_miles) of closest point
    in coords2 to each tuple in coords1.
    '''
    
    closest = []
    # for each pair of coordinates in coords1
    for firLoc in coords1:
        dis = []
        # calculate the distance to each coordinate pair in coords2
        for secLoc in coords2:
            # append the distance in miles
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
    
    #---------------------------------------------------------------------------
    # Find the biggest cluster of buildings
    #---------------------------------------------------------------------------
    '''
    There is a weather station that has the most buildings for which it is the
    closest weather station.  Determine this station and plot the energy use
    of these buildings.
    '''
    # list of all the indexes
    index_list = [x[0] for x in min_distance]

    # get the mode of the indexes (i.e., the most frequent)
    most_freq = max(set(index_list), key = index_list.count)

    # get the indices of the buildings that match the most frequent weather station
    keepers = []
    for index, tup in enumerate(min_distance):
        if tup[0] == most_freq:
            keepers.append(index)

    # subset the pandas dataframe to pick the relevant buildings and get their data
    keeperData = []
    for index in keepers:
        # get the site id
        siteId = metaData[u'SITE_ID'][index]

        # create path to data
        buiFullPath = os.path.join(os.pardir, os.pardir, 'csv-only', 'csv', '{}.csv'.format(siteId))

        # read energy data
        energy_kWh = np.genfromtxt(buiFullPath, delimiter=',',skip_header=1,
                                   usecols=2)

        # annual energy use in kBTU
        energyAnn_kBTU = np.sum(energy_kWh*3.412)
        
        # get meta data
        flrArea = metaData[u'SQ_FT'][index]
        industry = metaData[u'INDUSTRY'][index]

        # full building info
        buiInfo = (siteId, energyAnn_kBTU, flrArea, industry, buiFullPath)

        # save and append the data
        keeperData.append(buiInfo)
        
   
    #---------------------------------------------------------------------------
    # Final plotting of the data
    #---------------------------------------------------------------------------    
    '''
    Create a scatter plot with square footage on x axis, energy use on y axis
    and colored by industry type.
    '''
    # create a color for each of the unique industries
    indNames = set([x[3] for x in keeperData])
    numIndustries = len(indNames)

    # get a color from a color map
    cm = plt.get_cmap('Set1')
    colPts = np.linspace(0.0, 0.5, numIndustries)

    # relational database
    type_color_map = dict(zip(indNames, colPts))

    # get the data
    colors = [type_color_map[x[3]] for x in keeperData]
    sqFt = [x[2] for x in keeperData]
    eneUse = [x[1]/1000 for x in keeperData]
    areas = [np.interp(kk, [min(eneUse), np.percentile(eneUse, 25), 
                                    np.percentile(eneUse, 75),
                                    max(eneUse)],
                                    np.array([5, 10, 20, 40])*10) for kk in eneUse]
    # plot
    plt.scatter(sqFt, eneUse, c=colors, s = areas, edgecolor='')
    plt.xlabel('Square Feet')
    plt.ylabel('Annual Energy Use [MBTU]')

    
    
    
    # Ensure the required directory exists
    if not os.path.isdir('../../figures'):
        os.mkdir('../../figures')
    
    # Save figure
    plt.savefig('../../figures/buildingsdata-session3.png')
    

if __name__ == '__main__':
    main()
