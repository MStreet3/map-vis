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
    metaDataPath = os.path.join(parentDir, 'building_data',
                                'csv-only.tar', 'csv-only',
                                'meta', fname)
    metaData = pd.DataFrame.from_csv(metaDataPath,index_col=None)
    
    # locate the demand data
    demDataPath = os.path.join(parentDir, 'building_data',
                                'csv-only.tar', 'csv-only',
                                'csv')
    # locate the weather data
    weaDataPath = os.path.join(parentDir, 'weather')
    
    # locate tmy3 meta dat
    fname = 'TMY3_StationsMeta.csv'
    tmy3DataPath = os.path.join(parentDir, 'building_data',
                                'csv-only.tar', 'csv-only',
                                'meta', fname)
    tmy3MetaData = pd.DataFrame.from_csv(tmy3DataPath,index_col=None)   
    
    # reverse geocode the data
    lat = metaData[u'LAT'].values
    lng = metaData[u'LNG'].values
#    city = []
#    state = []
#    reversed_geocode = []
#    for pair in zip(lat, lng):
#        g = geocoder.google(list(pair), method = 'reverse')
#        city.append(g.city)
#        state.append(g.state)
#        reversed_geocode.append(g)
#    
#    # replace dataframe
#    metaData['City'] = city
#    metaData['State'] = state
    
    # Find closest TMY3 weather data
    min_distances = []
    tmy3_file = []
    tmy3_lat = tmy3MetaData['Latitude'].values
    tmy3_lng = tmy3MetaData['Longitude'].values
    for pair in zip(lat, lng):
        dis = []
        for tmy3_loc in list(zip(tmy3_lat, tmy3_lng)):
            dis.append(vincenty(pair, tmy3_loc).miles)
        dis = np.array(dis)
        min_loc = (np.min(dis), np.argmin(dis))
        
        min_distances.append(min_loc)
        tmy3_file.append([tmy3MetaData['Site Name'][min_loc[1]],
                          tmy3MetaData['State'][min_loc[1]],
                          tmy3MetaData['USAF'][min_loc[1]]])
    # all airport data
    airport_names = set([x[0] for x in tmy3_file])
    airport_USAF = set([x[2] for x in tmy3_file])
    criterion = tmy3MetaData['Site Name'].map(lambda x: x in airport_names)
    unique_tmy3 = tmy3MetaData[criterion]
    uniTmyLng = unique_tmy3['Longitude'].values
    uniTmyLat = unique_tmy3['Latitude'].values

    # analyze the most buildings that share same weather file
    most_frequent = np.bincount([x[1] for x in min_distances]).argmax()
    keepers = []
    for index, tup in enumerate(min_distances):
        if tup[1] == most_frequent:
            keepers.append(index)
    
    buildingMetas = metaData.iloc[keepers]
    buildingFileNames = ['{}.csv'.format(x) for x in buildingMetas.SITE_ID.values]
    buiFileNamFull = [os.path.join(demDataPath, fnam) for fnam in buildingFileNames]
    buiDfs = [pd.DataFrame.from_csv(fnam, index_col=None) for fnam in buiFileNamFull]
    
    # most frequent airport
    mostFreqAirport = tmy3MetaData.iloc[most_frequent]
    
    # get weather data
    filesInWeaDir = [f for f in os.listdir(weaDataPath) if os.path.isfile(os.path.join(weaDataPath, f))]
    epwFiles = [f for f in filesInWeaDir if '.epw' in f]
    mostFreqWeaFile = [f for f in epwFiles if str(mostFreqAirport.USAF) in f]
    mostFreqWeaDat = weather.epwFile(mostFreqWeaFile[0], weaDataPath)
    
    # get total radiation
    totalRad = mostFreqWeaDat.dirNorRad + mostFreqWeaDat.diffHorRad
    dryBulb = mostFreqWeaDat.dryTem
    windSpeed = mostFreqWeaDat.winSpe
    
    # interpolate the weather data
    stepPerHour = 12
    x = np.linspace(1, 8760, 8760*stepPerHour)
    xp = range(1, 8761)
    interpWindSpeed = np.interp(x=x, xp=xp, fp=windSpeed)
    interpTotRad = np.interp(x=x, xp=xp, fp=totalRad)
    interpDryBulb = np.interp(x=x, xp=xp, fp=dryBulb)   
    
    tau = 1/stepPerHour
    offPea = 6.5 # $/kW
    semiPea = 11.7 # $/kW
    peak = 16.4 # $/kW

    TOU = np.concatenate((np.ones(6)*offPea,
                    np.ones(5)*semiPea,
                    np.ones(7)*peak,
                    np.ones(6)*offPea))

    # exploit numpy's float indexing to simulate a zero order hold interpolation
    interpTOU = np.array([TOU[kk] for kk in np.linspace(0, len(TOU)-1, 
                          len(TOU)*stepPerHour)])
    
    x = np.arange(90*24*stepPerHour,97*24*stepPerHour)
    fullinterpTOU = np.tile(interpTOU, len(buiDfs[0].value.values[x])/len(interpTOU))    
    
    
    # plot the demand for each building
    demandVals = []
    allMetrics = []
    cm = plt.get_cmap('jet')
    cmVals = np.linspace(0.2, 1.0, len(buiDfs))
    f, (ax1, ax2) = plt.subplots(nrows=2,sharex=True)

    indexTime = buiDfs[0].dttm_utc
    for index, df in enumerate(buiDfs):
        demandVals.append(df.value.values[x])
        metrics = getCorrCoeff(df.value.values[x],
                             fullinterpTOU, 
                             interpTotRad[x],
                             interpDryBulb[x],
                             interpWindSpeed[x],stepPerHour)
        allMetrics.append(metrics)
        ax1.plot(x, df.value.values[x], linewidth=1.5, color=cm(cmVals[index]))
    
    
    demandVals = np.array(demandVals)
    aggDemand = np.sum(demandVals, axis=0)
    allMetrics.append(getCorrCoeff(aggDemand, fullinterpTOU, 
                             interpTotRad[x],
                             interpDryBulb[x],
                             interpWindSpeed[x], stepPerHour))
    ax2.plot(x, aggDemand, linewidth=1.5, color='k')
    ax1.grid('on')
    ax2.grid('on')
    ax2.set_title('Total Aggregate Demand')
    ax1.set_title('Individual Component Demand')
    ax1.set_ylabel('kW')
    ax2.set_ylabel('kW')
    ax1.set_xlim((x[0], x[-1]))
    plt.savefig('../figures/nasville-comm.png')
    
    f, (ax1,ax2,ax3) = plt.subplots(nrows=3,sharex=True)
    ax1.plot(x, aggDemand, linewidth=1.5,color='k')
    ax2.plot(x, fullinterpTOU, linewidth=1.5, color=cm(np.random.rand(1)[0]))
    ax1.set_ylabel('kW')
    ax1.set_title('Total Aggregate Demand')
    ax2.set_ylabel('$/kW')
    ax2.set_title('Demand Rate Tariff')
    ax3.plot(x, np.correlate(aggDemand, fullinterpTOU, 'same'),
             linewidth=1.5, color=cm(np.random.rand(1)[0]))
    ax3.set_title('Signal Cross-Correlation')
    ax1.grid('on')
    ax2.grid('on')
    ax3.grid('on')
    ax1.set_xlim((x[0], x[-1]))
    plt.savefig('../figures/demandcomparison-nashville.png')
    
    
    f, ax1 = plt.subplots()
    cmVals = np.linspace(0.2, 1.0, len(allMetrics))
    labels = ['Building {}'.format(x+1) for x in range(len(allMetrics[:-1]))]
    labels.append('Aggregate')
    cm = plt.get_cmap('gray')
    for index, results in enumerate(allMetrics):
        if index == len(allMetrics)-1:
            col = 'r'
            lwd = 3.0
        else:
            col = cm(cmVals[index])
            lwd=1.5

        ax1.plot(range(1, 5), results, linewidth=lwd,
             marker='o', color=col, label=labels[index])

    ax1.set_ylim((-1.0, 1.0))
    ax1.set_xlim((0.0, 5.0))
    ax1.legend(frameon=False, loc=0,ncol=2)
    ax1.set_xticks(range(1, 5))
    ax1.set_xticklabels(['LPCC', 'LSCC', 'LTCC', 'LWCC'])
    ax1.set_title('Pearson Correlation Coefficients')
    ax1.grid('on')
    plt.savefig('../figures/allmetrics.png')
    
        
    
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