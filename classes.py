# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 12:23:16 2016

@author: Mike
"""
from __future__ import division
import re
from tempfile import mkstemp
from shutil import move
import inspect
import numpy as np
import pdb
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
import pdb
import csv
import pandas as pd
import random
from scipy.stats import ks_2samp

class Building():
    '''
    A simple class definition that instantiates a building from electric load 
    data alone.
    '''
    
    def __init__(self, load=None, buildingType = None, loadType=None):
        assert isinstance(load, np.ndarray), \
        'Variable "load" is {}.  Must be of type numpy.ndarray.'.format(type(load))        
        self.load = load
        self.buildingType = buildingType
        self.loadType = loadType
    
    def choose(self, populationSize = 0, *args, **kwargs):
        '''
        Aggregate the loads of the population.
        '''
        # pdb.set_trace()
        return np.sum([self.generateLoad(*args,**kwargs) for kk in range(populationSize)],
                       axis = 0)
                       
    def generateLoad(self, alpha = 0.01):
        mean = self.load
        # pdb.set_trace()
        sd = alpha*mean
        newLoad = []
        for index, mu in enumerate(mean):
            s = np.random.randn(1)*sd[index] + mu
            newLoad.append(s)        
        return np.array(newLoad).flatten()
    
    #------------------------------------------------------------------------------
    #   Functions related to characterizing the demand
    #------------------------------------------------------------------------------  
        
    
    def decompose_daily(self):
        demand = self.load
        stepsPerHour = len(demand)//8760
        #pdb.set_trace()
        a = demand.reshape((-1, 24*stepsPerHour))
        means = np.mean(a, axis=1)
        #pdb.set_trace()
        result = []
        new_means = []
        for index in range(a.shape[0]):
            # pdb.set_trace()
            tmp = a[index, :] - means[index]
            new_means.append(np.ones(24*stepsPerHour)*means[index])
            result.append(tmp)
        # pdb.set_trace()
        return np.array(result).flatten(), np.array(new_means).flatten()
    
    def calc_FABD(self):
        new_demand, means = self.decompose_daily()
        # pdb.set_trace()
        numPositives = len(new_demand[new_demand > 0])
        posLoad = [x if x > 0.0 else 0.0 for x in new_demand]
        # posLoad[posLoad<0.0] = 0.0
        # pdb.set_trace()
        return numPositives/len(new_demand), new_demand, means, posLoad
    
    def calc_cv_ndiff(self):
        demand = self.load        
        firstdiff = np.diff(demand)
        normed = np.abs(firstdiff)/np.max(demand)
        s = np.std(normed)
        xbar = np.mean(normed)
        return s/xbar, normed
    
    def calc_R2_sol(self, solarRad):
        new_demand, means = self.decompose_daily()
        posLoad = np.array([x if x > 0.0 else 0.0 for x in new_demand])
        x = solarRad[solarRad > 0.0]
        y = posLoad[solarRad > 0.0]
        slope, intercept, r_value, p_value, std_err = linregress(x, 
                                                                 y)
        r_squared = pow(r_value, 2)
        return r_squared   
        
    def calc_R2_temp(self, dryTem):
        new_demand, means = self.decompose_daily()
        x = dryTem
        y = new_demand[:]
        slope, intercept, r_value, p_value, std_err = linregress(x[y>0], 
                                                                 y[y>0])
        r_squared = pow(r_value, 2)
        return r_squared                                                               
        
        

class EnergyPlusBuilding(Building):
    def __init__(self, filepath, load_column, buildingType, loadType, *args, **kwargs):
        ''' file to read should be a csv'''
        self.filepath = filepath
        self.dataframe = pd.read_csv(self.filepath, index_col = 0, *args, **kwargs)
        
        try:
            self.load_column = load_column
            # pdb.set_trace()
            if len(self.dataframe[load_column]) != 8760*6:
                self.load = self.dataframe[load_column].values[288:]
                self.loadPd = self.dataframe[load_column][288:]
            else:
                self.load = self.dataframe[load_column].values
                self.loadPd = self.dataframe[load_column]
        except KeyError:
            raise KeyError, 'The variable {} was not found in '.format(load_column) +\
                             'the columns list {}.'.format(self.dataframe.columns)
        
        self.buildingType = buildingType
          

class PopulationResultsAnalysis():    
    def __init__(self, popSize, controlDemandCurve, experimentalDemandCurves,
            _controlDemandCurves = None, _all_experimentalDemandCurves = None,
            _all_identifyBuildings = None, resultsDirectory = os.getcwd()):
        
        # get the population size
        self.popSize = popSize
        self.ks_analysis = False
        self.labels = ['Group {}'.format(x + 1) for x in range(self.popSize)]
        self.labels.insert(0, 'Control')
        self.cm = plt.get_cmap('PuBu')
        
        # assign inputs to attributes
        self.resultsDirectory = resultsDirectory
        self._experimentalDemandCurves = experimentalDemandCurves
        self._controlDemandCurves = _controlDemandCurves
        self._all_experimentalDemandCurves = _all_experimentalDemandCurves
        self._all_identifyBuildings = _all_identifyBuildings        
        #pdb.set_trace()        
        
        # validate the population size and data
        valid = experimentalDemandCurves.shape[0] == self.popSize
        assert valid, 'Data does not match given population size.'
        
        # assign each experiment to an attribute
        self.controlDemandCurve = controlDemandCurve
        fieldNames = ['experimentalDemandCurve_{}_building'.format(count + 1)
                      for count in range(popSize)]
        for index, name in enumerate(fieldNames):
            setattr(self, name, experimentalDemandCurves[index])

    def get_building_identities(self, sample_size = 1):
        ''' Get identities of all the buildings used to make the experimental
        load when sample_size number of experimental buildings were instantiated.'''        
        pass
    
    def get_decomposed_experimental_load(self, sample_size = 1):
        ''' Get the individual load curves used to compose a single experimental
        aggregate load where sample_size number of experimental buildings
        were instantiated.'''
        pass
    
    def get_aggregate_experimental_load(self, sample_size = 1):
        ''' Get the aggregate experimental load curve where sample_size number
        of experimental buildings were instantiated.'''
        pass
    
    def _full_ks_2sample(self):
        ''' Run 2 sample KS test on all the experimental aggregate loads.'''
        
        # ensure ks_test has not been run yet
        if not self.ks_analysis:
            self.D_stat = []
            self.p_value = []
            for load in self._experimentalDemandCurves:
                ks_test = ks_2samp(self.controlDemandCurve, load)
                self.D_stat.append(ks_test[0])
                self.p_value.append(ks_test[1])
            self.ks_analysis = True
        else:
            raise 'K-S test has already been completed for all experimental groups.'
    
            
    def get_ecdf(self, sample_size = 1):

        # validate the sample size
        valid = True
        if sample_size > self.popSize:
            valid = False
        if sample_size == 0:
            vector = getattr(self, 'controlDemandCurve')
        elif sample_size >= 1:
            vector = getattr(self, 'experimentalDemandCurve_' +\
                               '{}_building'.format(sample_size))
        else:
            valid = False
        if not valid:
            raise 'sample_size must be > 0 and < population size.'            

        # calculate the ECDF        
        uniques, counts = np.unique(vector, return_counts=True)
        x = np.cumsum(counts)
        y = np.sort(vector)[::-1] # sort and reverse the load data
        return x,y
        
    def _plot_all_p_values(self, title=''):
        if self.ks_analysis == True:
            fig = plt.figure(figsize=(14,10), dpi=150)
            ax = fig.add_subplot(111)
            x = np.arange(1, self.popSize + 1)
            ax.plot(x, self.p_value, color = self.cm(0.75), linewidth = 1.2,
                    marker = '.', markersize = 10.0, label = 'P-values')
            xlabels = self.labels[1:]
            xlabels.insert(0, '')
            xlabels.append('')
            ax.set_xticks(np.arange(0, self.popSize + 2))
            ax.set_xticklabels(xlabels)   
            ax.legend(frameon = False)
            fig.savefig('pvalues_popSize_{}.png'.format(self.popSize))
        else:
            raise ValueError, 'KS analysis has not been run yet.'

    def _dbg_control_curves(self, 
                            controlLoad, title = 'Debug Curves for Control Population'):
        fig = plt.figure(figsize=(14,10), dpi=150)
        ax = fig.add_subplot(111)
        cValues = np.linspace(0.5, 1.0, self.popSize + 1)
        labels = ['Building Realization {}'.format(x+1) for x in range(self.popSize)]

        for num in range(self.popSize):
            ax.plot(self._controlDemandCurves[num][:24*6]/10000, linewidth = 1.2,
                    color = self.cm(cValues[num]), label = labels[num] )
        
        ax.plot(controlLoad[:24*6]/10000, color = 'r', linewidth = 1.2, label = 'Control Load')
        ax.set_ylabel('Building Demand [$MW$]')
        ax.set_title(title)
        ax.legend(frameon=False, ncol=2)
        fig.savefig(os.path.join(os.getcwd(),
                "debug_controlCurves_popSize_{}".format(self.popSize)+".png"))
        
    def _plot_all_ecdf(self, title=''):
        fig = plt.figure(figsize=(14, 10), dpi=150)
        ax = fig.add_subplot(111)
        cValues = np.linspace(0.5, 1.0, self.popSize + 1)
        for num in range(self.popSize + 1):
            xc, yc = self.get_ecdf(num)
            ax.plot(xc/6, yc/10000, linewidth = 1.2, 
                    color = self.cm(cValues[num]), label=self.labels[num])
        ax.set_ylabel('Building Demand [$MW$]')
        ax.set_xlabel('Time [h]') 
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(os.getcwd(),
                                 "popSize_{}".format(self.popSize)+".png"))


class PopulationGenerator():
    ''' 
    Class of functions to generate a population of buildings for analysis.
    A population is a dictionary with key's corresponding to the 
    attributes of self.prototypes and values corresponding to the count of 
    each prototype.
    '''
    
    def __init__(self):
        '''
        Class constructor
        '''
        
        self.prototypes = {1:'SmallOffice',
                  2:'MediumOffice',
                  3:'LargeOffice',
                  4:'StandAloneRetail',
                  5:'StripMall',
                  6:'PrimarySchool',
                  7:'SecondarySchool',
                  8:'OutpatientHealthcare',
                  9:'Hospital',
                  10:'SmallHotel',
                  11:'LargeHotel',
                  12:'WareHouse',
                  13:'QuickServiceRestaurant',
                  14:'FullServiceRestaurant',
                  15:'MidRiseApartment',
                  16:'HighRiseApartment'}
    def get_prototypes(self):
        '''
        Return a list of the prototypes.  Accessor method.
        '''
        return self.prototypes.values()
    
    def pick_random_pop_from_prototypes(self, popSize):
        '''
        This function generates a random population of buildings given
        a desired population size.
        
        Inputs:
        self (PopulationGenerator)
        popSize (int) - desired number of buildings
        
        Outputs:
        counts (dict) - dictionary of keys from self.prototypes
        '''
        
        # instantiate the dictionary with a count of zero for each type                  
        counts = {}
        for buildingType in self.prototypes:
            counts[buildingType] = 0
        
        # randomly sample the building types to build the dictionary of counts    
        for kk in range(1, popSize+1):
            realization = random.sample(self.prototypes, kk)[0]
            buildingType = self.prototypes[realization]
            counts[buildingType] += 1        
        return counts
    
    def pick_specific_pop_from_prototypes(self, buildingTypes, typeCounts):
        # instantiate the dictionary with a count of zero for each type                  
        counts = {}
        for buildingType in self.prototypes.itervalues():
            counts[buildingType] = 0
        # set specific counts from list
        for index, buildingType in enumerate(buildingTypes):
            counts[buildingType] = typeCounts[index]        
        return counts
            
    def gen_building_pop(self, counts, controlLoads, experimentalLoads, *args, **kwargs):
        '''
        This method uses the counts dictionary and load data to generate the 
        corresponding aggregate load data for a population.
        
        Inputs:
        counts (dict) - keys are the building types and values are integer of 
                        required number of each type.
        controlLoads (dict) - keys are the building types and values are 
                              numpy.ndarray of the load associated for the control
                              version of the building type.
        experimentalLoads (dict) - keys are the building types and values are
                                   numpy.ndarray of the load associated with the
                                   experimental version of the building type
        Outputs:
        
        '''
        
        # ensure that the counts dictionary is valid
        popSize = sum([value for key, value in counts.iteritems()])
     
        # first instantiate a population of control and experimental buildings    
        controlPop = []
        experimentalPop = []
        for key in counts:
            # if there is more than zero of the current type requested
            if counts[key] > 0:
                # add a Building object to the list 
                for value in range(counts[key]):
                    # instantiate a control Building object referencing the controlLoads dictionary
                    controlPop.append(Building(load = controlLoads[key],
                                               buildingType = key,
                                               loadType = 'Control'))
                    # instantiate an experimental Building object referencing the experimentalLoads dictionary
                    experimentalPop.append(Building(load = experimentalLoads[key],
                                                    buildingType = key,
                                                    loadType = 'Experimental'))
    
        # create the control load data
        _controlDemandCurves = []
        for building in controlPop:
            # for each building in the control population generate a random version
            # of its load data with the choose method
            realization = building.choose(1, *args, **kwargs)
            
            # keep track of each of these random loads by saving in the private
            # list _controlDemandCurves
            _controlDemandCurves.append(realization)
        
        # aggregate the loads stored in the private list
        controlDemandCurve = np.sum(_controlDemandCurves, axis = 0)
        
        # create experimental demand curves and the control demand curve
        _all_identifyBuildings = [] # private list with all building identities
        _all_experimentalDemandCurves = []    # private list with all the experimental demand data
        
        experimentalDemandCurves = []
        for kk in range(1, popSize+1):
            # ---------------------------------------------------------------
            # get a new population by selecting kk random buildings to change
            # ---------------------------------------------------------------
            
            # indexes of buildings to replace with experimental versions
            index2change = random.sample(range(popSize), kk)
            
            # indexes of buildings to keep as control
            index2keep = list(set(range(popSize)) - set(index2change))
    
            # create a new population of building objects
            newPop = []
    
            for index in index2keep:
                # add the control buildings to the new list
                newPop.append(controlPop[index])
            
            for index in index2change:
                # add the experimental buildings to the new list
                newPop.append(experimentalPop[index])

            # append a tuple of identifying information to the private list
            # for later use.
            _identifyBuildings = [(building.buildingType,
                                   building.loadType,
                                   building.load) for building in newPop]

            # generate a load curve for each building in the new population
            _experimentalDemandCurves = []
            for building in newPop:
                realization = building.choose(1, *args, **kwargs)
                
                # save each random realization as a list
                _experimentalDemandCurves.append(realization)
                
            experimentalDemand = np.sum(_experimentalDemandCurves,
                                         axis=0)
            # save the aggregate load data
            # nested private list of identity information
            _all_identifyBuildings.append(_identifyBuildings)
            
            # nested private list of the load information
            _all_experimentalDemandCurves.append(_experimentalDemandCurves)
            # pdb.set_trace()

            experimentalDemandCurves.append(experimentalDemand)
        
        # convert type for consistency
        experimentalDemandCurves = np.array(experimentalDemandCurves)

        return (controlDemandCurve, experimentalDemandCurves,
                _controlDemandCurves, _all_experimentalDemandCurves,
                _all_identifyBuildings)

def test_replace():
    file_path = os.path.expanduser(os.path.join(
    '~', 'Dropbox', 'docs', 'diss-prep',
    'diss-proposal', 'sustainableProductionConsumptionPaper',
    'building_data', 'sample_lbc',
    'lbc_baseline_NS_bar_EW_mov_DC.idf'))
    pattern = '[\w\s]*Timestep,[\w\s]*\d;'
    substr = '  Timestep, 6;'
    replace(file_path, pattern, substr)
    with open(file_path, 'a') as myfile:
        myfile.write("Output:Variable,\n  *, \n  Facility Total Electric Demand Power, \n" + \
                     '  Timestep;')

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(re.sub(pattern, subst, line))
    os.close(fh)
    #Remove original file
    os.remove(file_path)
    #Move new file
    move(abs_path, file_path)

def add_idf_output():
    pass





def main():
    '''
    Test function to grab data from a control building and experimental building.
    Then I'll plot some ecdfs with a population size of 1 building and 15 
    buildings.
    '''
    #--------------------------------------------------------------------------
    # Set up the search for the data
    #--------------------------------------------------------------------------
    
    scriptName = inspect.getfile(inspect.currentframe())
    scriptDir = os.path.dirname(os.path.abspath(scriptName))
    buildingType = 'ApartmentMidRise'
    city = 'Memphis'
    years = [2004, 2013]
    names = ['_'.join(['ASHRAE90.1',buildingType,'STD{}'.format(year),city]) +\
             '.csv'for year in years]
    fullNames = [os.path.join(scriptDir, 'building_data',
                              '_'.join([buildingType,'STD{}'.format(years[index])]),
                              name) for index, name in enumerate(names)]
    load_column = 'Whole Building:Facility Total Electric Demand Power [W](TimeStep) '
    load_type = ['Control', 'Experimental']

    #--------------------------------------------------------------------------
    # Instantiate the population of buildings with the EnergyPlus data
    #--------------------------------------------------------------------------

    buildings = [EnergyPlusBuilding(filepath, load_column, 
                                    buildingType, load_type[index]) 
                                    for index, filepath in enumerate(fullNames)]

    # generate a building population class
    populationOne = PopulationGenerator()
    counts = populationOne.pick_specific_pop_from_prototypes([buildingType],
                                                             [1])
    popSize = np.sum(np.array(counts.values()))
    
    # create inputs to the experiment
    controlLoads = {buildingType:buildings[0].load}
    experimentalLoads = {buildingType:buildings[1].load}
    
    # complete the load curve aggregator
    (controlDemandCurve, experimentalDemandCurves,
     _controlDemandCurves, _all_experimentalDemandCurves,
     _all_identifyBuildings) = populationOne.gen_building_pop(counts, 
                                                              controlLoads, 
                                                            experimentalLoads)
                                                            
    #--------------------------------------------------------------------------
    # Analyze the population of building data
    #--------------------------------------------------------------------------   
    populationOneAnalysis = PopulationResultsAnalysis(popSize,controlDemandCurve,
                                              experimentalDemandCurves,
                                              _controlDemandCurves, 
                                              _all_experimentalDemandCurves,
                                              _all_identifyBuildings)    
    # Run the KS Test on data
    populationOneAnalysis._full_ks_2sample()
    
    # Plot the data
    xc, yc = populationOneAnalysis.get_ecdf(0)
    xe, ye = populationOneAnalysis.get_ecdf(popSize)

    plt.figure(figsize=(7,5), dpi=150)
    plt.plot(xc/6, yc/10000, linewidth = 1.2, color = 'r', label = 'ASHRAE90.1-2004')
    plt.plot(xe/6, ye/10000, linewidth = 1.2, color = 'b', label = 'ASHARE90.1-2013')
    plt.title(buildingType + '\n Population Size: {}'.format(popSize))
    plt.legend()
    plt.ylabel('Building Demand [$MW$]')
    plt.xlabel('Time [h]')
    plt.savefig(buildingType+'_popSize_{}.png'.format(popSize))
    
    populationOneAnalysis._plot_all_ecdf(title=buildingType)
    populationOneAnalysis._plot_all_p_values()
    populationOneAnalysis._dbg_control_curves(controlLoads[buildingType])
    # pdb.set_trace()

#if __name__ == '__main__':
#    test_replace()           