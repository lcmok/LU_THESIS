# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:02:22 2020

@author: lonem
"""

#THIS SCRIPT WILL CREATE THE DATA NEEDED FOR A CONTINGENCY TABLE/CONFUSION MATRIX. IT WILL NOT CONSTRUCT THE TABLE ITSELF, BUT RATHER GIVE YOU THE DATA NECESSARY IN NEWDATA, NEWDATA2 AND DATA.
#CHANGE THE THRESHOLDS ACCORDING TO THE ONES RELEVANT FOR YOUR LOCATION (CAN CALCULATE WITH THE SCRIPT FIG7_DEFINE_5YRTHRESHOLD))

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#%%

location = 'mwakimeme'
#Read in data
if location == 'chikwawa':
    data = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/DATA/ChikwawaContingency2.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
else:
    data = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/DATA/KarongaContingency2.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])

#Flood is boolean, depicts documented day of flood or not
data.columns = ['rcmc', 'm', 'Flood']
#data = data.drop(labels=['cmc', 'm'], axis=1)
data['doy']= data.index
data.index = pd.to_datetime(data.index, dayfirst=True)
#%% What signal do we want to look at
signal = 'rcmc'

#THESE THRESHOLDS WERE DEFINED FROM THE SCRIPT FIG7_DEFINE_5YRTHRESHOLD
#CHANGE ACCORDING TO LOCATION STUDIED

if location == 'chikwawa':
    if signal == 'rcmc':
        tres = 0.1680877186910525
    else:
        tres = 2.6587580509238578
if location == 'mwakimeme':
    if signal == 'rcmc':
        tres = 0.08099764762274506
    else:
        tres = 3.075676335347403
    
nans = 0
hit = 0
miss = 0
falsealarm = 0
correctnegatives = 0
firstmoment = 0


data["floodseason"] = data.index.year
data.loc[data.index.month < 5, "floodseason"] = data["floodseason"]-1  #will make the year run from May to April instead of January to December (MALAWI HYDROLOGICAL SEASON)
data["floodseason"] = pd.to_datetime(data["floodseason"], format='%Y')

data['modelled'] = 0
data['allmodeldays'] = 0
data['observed'] = np.nan


#LIST OF ALL DAYS INCL MODELLED OR OBSERVED FLOODS. CAN SCROLL THROUGH IT TO SEE IF MOD FALLS WITHIN A CERTAIN DAYS OF OBS.
data.loc[data[signal] >= tres, ['allmodeldays']] = 1

#NEWDATA = LIST OF THE FIRST FLOOD DAY OF EACH FLOOD SEASON (MODELLED)
newdata = data[data[signal] >= tres]
newdata = newdata.drop_duplicates(subset=['floodseason'])
newdata.modelled=1

#NEWDATA2 = LIST OF THE FIRST FLOOD DAY OF EACH FLOOD SEASON (FROM DATABASE)
newdata2 = data[data['Flood'] == 1]
newdata2 = newdata2.drop_duplicates(subset=['floodseason'])
newdata2.observed=1

#merge in dataset
for index, row in newdata.iterrows():
    data.loc[data.doy == row.doy, ['modelled']] = 1
    
for index, row in newdata2.iterrows():
    data.loc[data['doy'] == row['doy'], ['observed']] = 1
    
#%%
plt.plot(Magf[0])
plt.plot(Magf.index, [tres]*len(Magf))    
