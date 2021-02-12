# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:32:44 2020

@author: lcmok
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:29:05 2020

@author: lcmok
"""
#THIS MAIN SCRIPT WILL LET YOU EXTRACT THE TB VALUES FROM THE .NC FILES ON YOUR PC. PLEASE ENSURE THE .NC FILES ARE IN YOUR WORKING DIRECTORY AND ARE NAMED AS FOLLOWS:
#A_YYY.nc -> if you use several platforms within one year, name the next platform B, and so forth.
#The .nc files can be downloaded using the software Wget (Linux/Apple/Windows) and the DownloadSatdata_LM script. Read the documentation there.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy
from math import factorial
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import statsmodels.formula.api as smf
from numpy import polyfit
from scipy import signal
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

#TO CHANGE:
location = 'chikwawa' #chikwawa or mwakimeme
#Add years of interest in chronological order
years = list(range(1978, 2018))


#--------

#The downloaded data is in the projection cylindrical equal area
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
with open('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Scripts/poi_'+location+'.txt') as f: #change string to your own location, this refers to a txt file with the coordinate sets. An example for cell C0 is 34.88398, -15.99103
    points = [tuple(map(float, i.split(','))) for i in f]
    points_xy = nsidc2.proj_coords(points, proj_out, proj_in)
    #Unpack points in coordinate system of netcdf
    points_x, points_y = zip(*points_xy)
    
with open('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Scripts/wcpoi_poi.txt') as f: #idem dito
    wcpoints = [tuple(map(float, i.split(','))) for i in f]
    wcpoints_xy = nsidc2.proj_coords(wcpoints, proj_out, proj_in)
    #Unpack points in coordinate system of netcdf
    wcpoints_x, wcpoints_y = zip(*wcpoints_xy)   

#-------------------------#---------------------------#---------------------------#---------------------------#
# CALCULATION OF RATIOS FOR 1 LOCATION ONLY (for several locations, please use the script in the Fig8_TLCC script)
#---------------------------#---------------------------#---------------------------#---------------------------#

#note that for karonga, the downstream poi was the second one in the list due to some problems with the signal. this is why it starts with number 2

# CM
df, Magf, Cf, Mf, Ccoord, Mcoord, timeseries, ds = nsidc2.calc_ratio1pt(years, points_x, points_y,location)
# CMC
cmcdf, cmcCcoord, cmcMcoord, ds2, cmcwet, Cwf = nsidc2.calc_cmcratio1pt(years, points_x, points_y, wcpoints_x, wcpoints_y, timeseries,location)

#%%
#-------------------------#---------------------------#---------------------------#---------------------------#
#DATA CORRECTION/TREATMENT AND SUBSETTING FOR DISCHARGE
#---------------------------#---------------------------#---------------------------#---------------------------#

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
    poi = 0
elif location == 'mwakimeme':
    dis.index = mwakitime
    poi = 2

#%% REMOVE AWAY DRY SEASON FROM ORIGINAL DATA
cmcdf[(cmcdf.index.month>4) & (cmcdf.index.month<12)] = np.nan
Magf[(Magf.index.month>4) & (Magf.index.month<12)] = np.nan
dis[(dis.index.month>4) & (dis.index.month<12)] = np.nan

#%%-------------------------#---------------------------#---------------------------#---------------------------#
#DATA PLOTTING: CM-ratio and discharge
#---------------------------#---------------------------#---------------------------#---------------------------#

# Create the general figure
fig = plt.figure(figsize=(14,6))
# Plot mwaki data
ax1 = fig.add_subplot(111)
plt.title='Discharge and satellite signals in Chikwawa'
plt.xlabel('Day of Year')
ax1.plot(Magf[poi], linestyle='-', color='b', label = 'Flood Magnitude')     
ax1.set(ylabel="Flood magnitude")
# Add filtered data in the same figure
ax2 = fig.add_subplot(111, frameon=False)
ax2.plot(dis, linestyle='-', color='r', label = 'Discharge')
ax2.yaxis.set_label_position("right")
ax2.set(ylabel="Discharge (m3/s)")
ax2.yaxis.tick_right()

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines = lines_1 + lines_2
labels = labels_1 + labels_2

ax1.legend(lines, labels, loc=0)
 
#%%-------------------------#---------------------------#---------------------------#---------------------------#
#DATA PLOTTING: 2D-brightness temperatures in chikwawa
#---------------------------#---------------------------#---------------------------#---------------------------#
plt.figure(figsize=(17,12)) 
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.8)
ax.add_feature(cartopy.feature.RIVERS, linestyle='-', alpha=.8)
xi, yi =np.meshgrid(ds.x, ds.y)
Mcoord_xy = nsidc2.proj_coords(Mcoord, proj_in, proj_out)
Ccoord_xy = nsidc2.proj_coords(Ccoord, proj_in, proj_out)

if location == 'chikwawa':
    wcpointsnew = ds.sel(x=wcpoints_x[0], y=wcpoints_y[0], method='nearest')
    wcx = float(wcpointsnew.x.values)
    wcy = float(wcpointsnew.y.values)
    wcpointsnew = [(wcx, wcy)]
    wcpointsnew = nsidc2.proj_coords(wcpointsnew, proj_in, proj_out)
    #to ensure karonga and chikwawa can be plotted in one plot
    wcpointsnewsave = wcpointsnew
    Ccoordsave = Ccoord_xy
    Mcoordsave = Mcoord_xy

else:
    wcpointsnew = ds.sel(x=wcpoints_x[1], y=wcpoints_y[1], method='nearest')
    wcx = float(wcpointsnew.x.values)
    wcy = float(wcpointsnew.y.values)
    wcpointsnew = [(wcx, wcy)]
    wcpointsnew = nsidc2.proj_coords(wcpointsnew, proj_in, proj_out)

if location == 'chikwawa':
    loni, lati = pyproj.transform(proj_in, proj_out, xi, yi)
    p = ax.pcolormesh(loni, lati, ds['TB'].values[5], transform=ccrs.PlateCarree(), cmap='terrain')
    # also plot some points of interest  -> note that this plots both locations in one plot, if the script does not not Ccoordsave and Mccoordsave for example, this is because you are working with one location
    #in that case, please delete the lines below. If you are working with both, please run 'chikwawa' first and then 'mwakimeme'. In that way, the points will be plotted on the same map.
    ax = nsidc2.plot_points(ax, Ccoordsave, marker='o', color='r', linewidth=0., transform=ccrs.PlateCarree())
    ax = nsidc2.plot_points(ax, Mcoordsave, marker='o', color='g', linewidth=0., transform=ccrs.PlateCarree())
    ax = nsidc2.plot_points(ax, wcpointsnewsave, marker='o', color='k', linewidth=0., transform=ccrs.PlateCarree())
else:
    ax = nsidc2.plot_points(ax, Ccoord_xy, marker='o', color='r', linewidth=0., transform=ccrs.PlateCarree())
    ax = nsidc2.plot_points(ax, Mcoord_xy, marker='o', color='g', linewidth=0., transform=ccrs.PlateCarree())
    ax = nsidc2.plot_points(ax, wcpointsnew, marker='o', color='k', linewidth=0., transform=ccrs.PlateCarree())

plt.colorbar(p, label='Brightness temperature [K]')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle=':')
plt.title = ('Brightness Temperature')
gl.xlabels_top = False
gl.ylabels_right = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER

# Gracefully close ds
ds.close()
#plt.savefig('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Scripts/Figures/Tb_map.png', dpi = 400)

#%%-------------------------#---------------------------#---------------------------#---------------------------#
#COPYING DATA FOR IN EXCEL
#---------------------------#---------------------------#---------------------------#---------------------------#
# Selecting data for excel -> COPIES TO CLIPBOARD, Cd, M and Cw --> will be used in Fig2_raw_tb.py
f = pd.concat([pd.DataFrame(Cf[poi]), pd.DataFrame(Mf[poi]), pd.DataFrame(cmcwet, index=timeseries)], axis=1)
f.to_clipboard()

#%% -> COPIES TO CLIPBOARD, Q, cmc and m and discharge --> will be used in Fig3_4_Tb_vs_discharge.py
f = pd.concat([pd.DataFrame(dis), pd.DataFrame(cmcdf[poi]), pd.DataFrame(Magf[poi]),], axis=1)
f.to_clipboard()
