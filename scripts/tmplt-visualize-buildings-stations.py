# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 15:07:49 2016

@author: Mike
"""

# import potentially useful packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import inspect
from geopy.distance import vincenty
from mpl_toolkits.basemap import Basemap

def findClosestStation(coords1, coords2):
    '''
    Function finds the closest value of coords2 to each value in coords1.
    Inputs:
    coords1 (iterable) - contains (lat, long) pairs
    coords2 (iterable) - contains (lat, long) pairs
    
    Outputs:
    closest (list) - contains index value of coords2 that corresponds to the 
    closest (lat, long) pair for each index in coords1
    '''
    
    # determine the distance from each pair in coords1 to each pair in coords2

    # store the index of coords2 corresponding to the minimum distance     
    
    pass

def main():
    '''
    This function plots the locations of 100 commercial buildings across 
    the United States and the closest weather station to each building.
    
    A file with (lat, long) data for 100 buildings is located at
    '../csv-only/meta/all_sites.csv'
    
    A file with (lat, long) data for several hundred weather stations is located
    at '../csv-only/meta/TMY3_StationsMeta.csv'
    '''
    #--------------------------------------------------------------------------
    # Locate the data files
    #--------------------------------------------------------------------------
    # locate the meta data file
    # get the current file location

    
    # locate the building meta data
    
    # locate the weather data
    
    # locate tmy3 weather meta data

    #--------------------------------------------------------------------------       
    # Read the relevant lat,long data into memory
    #--------------------------------------------------------------------------



    #--------------------------------------------------------------------------
    # determine the relevant weather stations
    #--------------------------------------------------------------------------
    # Write a function that returns the weather station
    # closest to each building.

    #--------------------------------------------------------------------------
    # Plotting
    #--------------------------------------------------------------------------
    # Plot a map of the united states using Basemap.
    # Plot only the weather stations closest to the buildings.
    # Save the file in '../figures'

    

if __name__ == '__main__':
    main()