# -*- coding: utf-8 -*-
"""
Created on Fri Jan 01 19:18:27 2016

@author: Mike
"""

import os
import csv
import numpy as np

class epwFile():
    '''
    Class for instantiating and manipulating weather data in EPW format.
    
    '''
    def __init__(self, fileNam, path=os.getcwd()):
        self.filepath = os.path.join(path, fileNam)
        self.year = [] # not used typically
        self.month = [] # 1-12
        self.day = [] # depends on month
        self.hour = [] # 1-24. Hour 1 is 00:01 to 01:00
        self.minute = [] # 1 - 60
        self.uncFlags = []
        self.dryTem = [] # C
        self.dewTem = [] # C
        self.relHum = [] # [ ]
        self.staPre = [] # Pa
        self.extHorRad = [] # Wh/m2
        self.extDirNorRad = [] # Wh/m2
        self.HorInfRadInt = [] # Wh/m2
        self.gloHorRad = [] # Wh/m2
        self.dirNorRad = [] # Wh/m2
        self.diffHorRad = [] # Wh/m2
        self.gloHorIll = [] # lux
        self.dirNorIll = [] # lux
        self.diffHorIll = [] # lux
        self.zenLum = [] # Cd/m2
        self.winDir = [] # degrees
        self.winSpe = [] # m/s
        self.totSkyCov = [] # fraction
        self.opaSkyCov = [] # fraction
        self.vis = [] # km
        self.ceiHei = [] # m
        self.preWeaObs = []
        self.preWeaCodes = []
        self.preWat = [] # mm
        self.aerOptDep = [] # thousandths
        self.snoDep = [] # cm
        self.daySinceLasSno = [] # days
        self.albedo = [] # fraction
        self.liqPreDep = [] # mm
        self.liqPreQuantity = [] # hr          
        with open(self.filepath, 'rb') as epwFile:
            reader = csv.reader(epwFile, delimiter=',')
            rowNum = 0
            for row in reader:
                if rowNum == 0:
                    # LOCATION
                    self.city = row[1]
                    self.state = row[2]
                    self.country = row[3]
                    self.weaDatTyp = row[4]
                    self.wmo = row[5]
                    self.lat = float(row[6])
                    self.longitude = float(row[7])
                    self.tz = float(row[8])
                    self.elevation = float(row[9]) # meters above sea level
                elif rowNum == 1:
                    # DESIGN CONDITIONS
                    # The Design Conditions header record encapsulates matching design
                    # conditions for a location.  Currently only those design conditions
                    # contained in the ASHRAE Handbook of Fundamentals 2009 are contained
                    # in the weather files.  
                    self.conditions = ','.join(row)
                elif rowNum == 2:
                    # TYPICAL/EXTREME PERIODS
                    self.periods = ','.join(row)
                elif rowNum == 3:
                    # GROUND TEMPERATURES
                    self.groundTmps = ','.join(row)
                elif rowNum == 4:
                    # HOLIDAYS/DAYLIGHT SAVING
                    self.leapYear = row[1]
                    self.holidays = ','.join(row)
                elif rowNum in [5,6]:
                    # Skipping the comment lines
                    pass
                elif rowNum == 7:
                    # DATA PERIODS
                    self.dataPeriods = int(row[1])
                    self.recordsPerHour = int(row[2])
                    fieldNames = ['period'+ str(x + 1) + y 
                                   for x in range(self.dataPeriods) 
                                   for y in ['_Description',
                                             '_StartDayOfWeek',
                                             '_DataPeriodStart',
                                             '_DataPeriodEnd']]
                    for index, fieldName in enumerate(fieldNames):
                        setattr(self, fieldName, row[index + 3])
                    
                else:
                    self.year.append(row[0]) # not used typically
                    self.month.append(row[1]) # 1-12
                    self.day.append(row[2]) # depends on month
                    self.hour.append(row[3]) # 1-24. Hour 1 is 00:01 to 01:00
                    self.minute.append(row[4]) # 1 - 60
                    self.uncFlags.append(row[5])
                    self.dryTem.append(row[6]) # C
                    self.dewTem.append(row[7]) # C
                    self.relHum.append(row[8]) # [ ]
                    self.staPre.append(row[9]) # Pa
                    self.extHorRad.append(row[10]) # Wh/m2
                    self.extDirNorRad.append(row[11]) # Wh/m2
                    self.HorInfRadInt.append(row[12]) # Wh/m2
                    self.gloHorRad.append(row[13]) # Wh/m2
                    self.dirNorRad.append(row[14]) # Wh/m2
                    self.diffHorRad.append(row[15]) # Wh/m2
                    self.gloHorIll.append(row[16]) # lux
                    self.dirNorIll.append(row[17]) # lux
                    self.diffHorIll.append(row[18]) # lux
                    self.zenLum.append(row[19]) # Cd/m2
                    self.winDir.append(row[20]) # degrees
                    self.winSpe.append(row[21]) # m/s
                    self.totSkyCov.append(row[22]) # fraction
                    self.opaSkyCov.append(row[23]) # fraction
                    self.vis.append(row[24]) # km
                    self.ceiHei.append(row[25]) # m
                    self.preWeaObs.append(row[26])
                    self.preWeaCodes.append(row[27])
                    self.preWat.append(row[28])# mm
                    self.aerOptDep.append(row[29]) # thousandths
                    self.snoDep.append(row[30]) # cm
                    self.daySinceLasSno.append(row[31]) # days
                    self.albedo.append(row[32]) # fraction
                    self.liqPreDep.append(row[33]) # mm
                    self.liqPreQuantity.append(row[34]) # hr 
                            
                rowNum += 1

        # convert all list data into numpy arrays
        for attribute in self.__dict__.keys():
            # ensure the attribute is not a string
            if not isinstance(getattr(self, attribute), basestring):
                # skip the uncertainty codes
                if attribute != 'uncFlags':
                    # get the attribute then set it as numpy array with floats
                    setattr(self, attribute, 
                            np.array(getattr(self, attribute), 
                                     dtype=np.float))
  

