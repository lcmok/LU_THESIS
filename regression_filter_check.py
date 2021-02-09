# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#this script checks if the regression results improve when just non-filled cells (using the avgfill method) are used.
#ensure the nsidc script and the .nc files, as well as the poi txt file are in the same folder as this script is.

#For discharge
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy
from math import factorial
#import hydroeval #for NSE
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import statsmodels.formula.api as smf
from numpy import polyfit
from scipy import signal
#For satellite
import datetime
import pyproj
import xarray as xr
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from pandas.core.common import flatten
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa as smtsa
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
#import own library
import nsidc2

#%%
#-------------------------#---------------------------#---------------------------#---------------------------#
#VARIABLES (CHANGE WHERE APPROPIATE)
#---------------------------#---------------------------#---------------------------#---------------------------#
location = 'mwakimeme' #chikwawa or mwakimeme
#Add years of interest in chronological order
years = list(range(1978, 2018))
#From downloaded data: cylindrical equal area
proj4str = '+proj=cea +lat_0=0 +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m'
#Projection object for the projection used in the downloaded grids.
proj_in = pyproj.Proj(proj4str)
#Target: WGS84 (EPSG code 4326)
proj_out = pyproj.Proj(init='epsg:4326')       #unit is degrees

#Rough bounds of Malawi (y, x / long, lat) lower left to upper right
bounds_xy = [(-18.,  30.55),
          ( -8.3, 37.,),
         ]
#EXAMPLE OF POIS: 33.84683, -9.954743         
with open('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Scripts/poi_'+location+'.txt') as f:
    points = [tuple(map(float, i.split(','))) for i in f]
    points_xy = nsidc2.proj_coords(points, proj_out, proj_in)
    #Unpack points in coordinate system of netcdf
    points_x, points_y = zip(*points_xy)
    
with open('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Scripts/wcpoi_poi.txt') as f:
    wcpoints = [tuple(map(float, i.split(','))) for i in f]
    wcpoints_xy = nsidc2.proj_coords(wcpoints, proj_out, proj_in)
    #Unpack points in coordinate system of netcdf
    wcpoints_x, wcpoints_y = zip(*wcpoints_xy)   

#%%
#-------------------------#---------------------------#---------------------------#---------------------------#
# CALCULATION OF RATIOS FOR 1 LOCATION ONLY (for several locations, please use the script in the Fig8_TLCC script)
#---------------------------#---------------------------#---------------------------#---------------------------#

#note that for karonga, the downstream poi was the second one in the list due to some problems with the signal. this is why it starts with number 2

# CM
df, Magf, Cf, Mf, df2, Magf2, Ccoord, Mcoord, timeseries, ds = nsidc2.calc_ratio1ptnew(years, points_x, points_y,location)
# CMC
cmcdf, cmcCcoord, cmcMcoord, ds2, cmcwet, Cwf, cmcdf2 = nsidc2.calc_cmcratio1ptnew(years, points_x, points_y, wcpoints_x, wcpoints_y, timeseries,location)

#%% Compare original and filled version and drop nan vals
cmcdfcompare = pd.concat([cmcdf, cmcdf2], axis=1).dropna()
cmcdfcompare.columns=['usedcmc', 'unfilteredcmc']
cmcdf = cmcdfcompare.drop(['unfilteredcmc'], axis=1)
magfcompare = pd.concat([Magf, Magf2], axis=1).dropna()
magfcompare.columns=['usedm', 'unfilteredm']
Magf = magfcompare.drop(['unfilteredm'], axis=1)
#%%discharge
#Load discharge data and recognize nan values
if location == 'chikwawa':
    chik = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/DATA/FinalLocationsFormatted/ShireChikwawa.csv', delimiter = ',', header = 0, na_filter=True)
    chik.columns = ['doy', 'Discharge']
    chik['doy'] = pd.to_datetime(chik.doy, dayfirst=True)
    chikss, chiktime = nsidc2.treatmentfunc(chik, years)
    discharge = np.asarray(chikss)
    time = np.asarray(chiktime)
elif location == 'mwakimeme':
    mwaki = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/DATA/FinalLocationsFormatted/NorthRukuruMwakimeme.csv', delimiter = ';', header = 0, na_values = -999, na_filter=True)
    mwaki['Date'] = pd.to_datetime(mwaki.Date, dayfirst=True)
    mwaki.columns = ['doy','Discharge']
    mwakiss, mwakitime = nsidc2.treatmentfunc(mwaki, years)
    discharge = np.asarray(mwakiss)
    time = np.asarray(mwakitime)
    
dis = pd.DataFrame(discharge)
if location == 'chikwawa':
    dis.index = chiktime
elif location == 'mwakimeme':
    dis.index = mwakitime
    
    #%%
#train test split
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

#Cut out last 5 years of data
if location == 'chikwawa':
    cmcdf = cmcdf[(cmcdf.index.year>2004) * (cmcdf.index.year<2010)]
    dis = dis[(dis.index.year>2004) * (dis.index.year<2010)]
    Magf = Magf[(Magf.index.year>2004) * (Magf.index.year<2010)]
else:
    cmcdf = cmcdf[(cmcdf.index.year>1985)* (cmcdf.index.year<1991)]
    dis = dis[(dis.index.year>1985) * (dis.index.year<1991)]
    Magf = Magf[(Magf.index.year>1985) * (Magf.index.year<1991)]
    
#DEPENDENT VARIABLE: SAT SIGNAL (X)
#INDEPENDENT VARIABLE: DISCHARGE (Y)
if location == 'mwakimeme':
    poi=2
else:
    poi=0

array = pd.concat([cmcdf['usedcmc'],dis, Magf['usedm']],axis=1).dropna()
array.columns = ['cmc', 'Q', 'mag']

newarray = np.asarray(array)

###### SIMPLE TRAIN-TEST SPLIT#######
#x = cmc, y = discharge, xm = m. Everything with a t next to it is the test data
x, xt, y, yt, xm, xmt = train_test_split(newarray[:,0], newarray[:,1], newarray[:,2], test_size=0.5, train_size=0.5)

# SAVE TRAINING AND TESTING DATASETS FOR FUTURE REFERENCE
traincmcdf = pd.DataFrame([x, y]).transpose()
testcmcdf = pd.DataFrame([xt, yt]).transpose()
trainmdf = pd.DataFrame([xm, y]).transpose()
testmdf = pd.DataFrame([xmt, yt]).transpose()

#%% APPLY CORRELATION AND CREATE MODEL
 
predictioncmc, rsqcmc, spcmc, modelcmc, polycmc, cmcx2, cmcy2 = nsidc2.createmodel(x,y, xt, yt, 2)
print('rsquared cmc')
print(rsqcmc)
print('rho cmc')
print(spcmc)

# #Test normality
# pval, resids = normal_errors_assumption(modelcmc, xt, yt, predictioncmc, p_value_thresh=0.05)
# resids=resids.reset_index(drop=True)
# #Test homoscedasticity
# homoscedasticity_assumption(modelcmc, xt, yt, predictioncmc)

#plt.plot(modelcmc)
predictionm, rsqm, spm, modelm, polym, mx2, my2 = nsidc2.createmodel(xm, y, xmt, yt, 2)
print('rsquared m')
print(rsqm)
print('rho m')
print(spm)

# #Test normality
# pval, resids = normal_errors_assumption(modelm, xmt, yt, predictionm, p_value_thresh=0.05)
# resids=resids.reset_index(drop=True)
# #Test homoscedasticity
# homoscedasticity_assumption(modelm, xmt, yt, predictionm)