# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:42:50 2021

@author: lcmok
"""
#THIS SCRIPT WILL DOWNLOAD THE NECESSARY DATA AND THEN PERFORM A TIME LAGGED CROSS CORRELATION. YOU WILL NEED TEXT FILES WITH YOUR POINTS OF INTEREST.
#IT PERFORMS IT ON BOTH RCMC AND M, AND FOR TWO LOCATIONS, AND PLOTS THEM TOGETHER IN ONE FIGURE. REMOVE THE LOOPS IF YOU ONLY NEED ONE INDEX OR ONE LOCATION.

#For discharge
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import hydroeval #for NSE
from scipy import signal
#For satellite
import pyproj
#import own library
import nsidc2

#%%
#-------------------------#---------------------------#---------------------------#---------------------------#
#VARIABLES
#---------------------------#---------------------------#---------------------------#---------------------------#

for location in ['chikwawa','mwakimeme']:
    #Add years of interest in chronological order
    years = list(range(1978, 2018))
    
    #-------------------------#---------------------------#---------------------------#---------------------------#
    #SETTINGS THAT PROBABLY WON'T NEED TO BE CHANGED + EXTRACTION OF TB, RATIOS
    #---------------------------#---------------------------#---------------------------#---------------------------#
    
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
    
    #-------------------------#---------------------------#---------------------------#---------------------------#
    # CALCULATION OF RATIOS
    #---------------------000000000000------#---------------------------#---------------------------#---------------------------#
    # CM
    df, Magf, Cf, Mf, Ccoord, Mcoord, timeseries, ds = nsidc2.calc_ratio(years, points_x, points_y,location)
    # CMC
    cmcdf, cmcCcoord, cmcMcoord, ds2, cmcwet, Cwf = nsidc2.calc_cmcratio(years, points_x, points_y, wcpoints_x, wcpoints_y, timeseries,location)

    #-------------------------#---------------------------#---------------------------#---------------------------#
    #DATA CORRECTION/TREATMENT AND SUBSETTING FOR DISCHARGE
    #---------------------------#---------------------------#---------------------------#---------------------------#
    
    #Load discharge data and recognize nan values
    if location == 'chikwawa':
        chik = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/DATA/FinalLocationsFormatted/ShireChikwawa.csv', delimiter = ',', header = 0, na_filter=True)
        chik.columns = ['doy', 'Discharge']
        chik['doy'] = pd.to_datetime(chik.doy, dayfirst=True)
        chikss, chiktime = nsidc2.treatmentfunc(chik, years)
        time = np.asarray(chiktime)
        k=0
    elif location == 'mwakimeme':
        mwaki = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/DATA/FinalLocationsFormatted/NorthRukuruMwakimeme.csv', delimiter = ';', header = 0, na_values = -999, na_filter=True)
        mwaki['Date'] = pd.to_datetime(mwaki.Date, dayfirst=True)
        mwaki.columns = ['doy','Discharge']
        mwakiss, mwakitime = nsidc2.treatmentfunc(mwaki, years)
        time = np.asarray(mwakitime)
        k=2
        
    #MAKE STATIONARY
    cmcstat, cmctrd, cmccurve = nsidc2.makestationary2(cmcdf, 2)
    magstat, magtrd, magcurve = nsidc2.makestationary2(Magf, 2)
    #ONLY LOOK AT WET SEASON
    cmcstat[(cmcstat.index.month>4) & (cmcstat.index.month<12)] = np.nan
    magstat[(magstat.index.month>4) & (magstat.index.month<12)] = np.nan
    
    for sig in ['rcmc', 'm']:
        #CHOOSE SIGNAL
        if sig == 'rcmc':
            frame = cmcstat
        else:
            frame = magstat
        
        #START OF LOOP TO CREATE TLCC
        corrlist = pd.DataFrame()
        poilist = []
            
        for i in [x for x in range(k, len(points_xy))]:
            lags = []
            print(i)
            poi = 'poi'+str(i)                  #Point of interest number (0 = downstream)
            poilist.append(poi)
            shifted = pd.DataFrame(frame[str(k)]) #New dataframe with just the original cm(c) or mag-data at the downstream poi
            lags.append(poi)
            collist = [poi]                     #For generating column headers
            
            for n in range(-20,21):
                dfnew = frame[str(i)].shift(n)  #Shift satellite data at poi
                dfnew.columns = [str(n)]
                shifted = pd.concat([shifted, dfnew], axis=1)
                lags.append(n) #For generating x axis
                
            corrmatrix = shifted.corr(method='spearman') #Correlate shifted data at poi with downstream discharge or cm
            corrmatrix.columns = lags
            corrmatrix.index = lags
            #Take the first column of matrix because this column shows how it downstream values correllate with upstream pois at lags
            templist = list(corrmatrix[poi])
            corrlist = pd.concat([corrlist, pd.DataFrame(templist[1:])], axis=1)
        lags = lags[1:]
        corrlist = corrlist.transpose()    
        corrlist.index = poilist
        corrlist.columns = lags
        #Labels for legend
        lbC = 'C'
        lbK = 'K'    
        maxlist = []
        ct = k
        cmap = plt.get_cmap('tab20')
        #save everything for plotting
        if location == 'chikwawa':
            if sig == 'rcmc':
                Cmaxlistcmc = maxlist
                Ccorrlistcmc = corrlist
            else:
                Cmaxlistm = maxlist
                Ccorrlistm = corrlist
        elif location == 'mwakimeme':
            if sig == 'rcmc':
                Kmaxlistcmc = maxlist
                Kcorrlistcmc = corrlist
            else:
                Kmaxlistm = maxlist
                Kcorrlistm = corrlist
#%%
#Save maxima for in table
for index, row in Ccorrlistcmc.iterrows():
    maxx = row.max()
    Cmaxlistcmc.append((row.loc[row == maxx].index[0], maxx))
for index, row in Ccorrlistm.iterrows():
    maxx = row.max()
    Cmaxlistm.append((row.loc[row == maxx].index[0], maxx))
for index, row in Kcorrlistcmc.iterrows():
    maxx = row.max()
    Kmaxlistcmc.append((row.loc[row == maxx].index[0], maxx))
for index, row in Kcorrlistm.iterrows():
    maxx = row.max()
    Kmaxlistm.append((row.loc[row == maxx].index[0], maxx))
    
CN = len(Cmaxlistcmc)
KN = len(Kmaxlistcmc)
#%% PLOT FIGURE
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, dpi=300, figsize=(7.08, 3.3))
#Plot 1: chikwawa rcmc
ct = 0
for index, row in Ccorrlistcmc.iterrows():
    color = cmap(float(ct-2)/CN)
    ax1.plot(lags,row, label=lbC+str(ct), c=color, linewidth=2)
    ct = ct+1
    
ax1.set_ylabel("Spearman's ρ",size=8)
ax1.set_title('A.', size=8)

#Plot 2: chikwawa m  
ct = 0  
for index, row in Ccorrlistm.iterrows():
    color = cmap(float(ct-2)/CN)
    ax2.plot(lags,row, label=lbC+str(ct), c=color,linewidth=2)
    ct = ct+1
    
ax2.set_title('B.', size=8)    

#Plot 3: karonga rcmc
ct = 0
for index, row in Kcorrlistcmc.iterrows():
    ax3.plot(lags,row, label=lbK+str(ct),linewidth=2)
    ct = ct+1
    
ax3.set_xlabel('Lag time (days)', size=8)
ax3.set_ylabel("Spearman's ρ", size=8)
ax3.set_title('C.', size=8)

#Plot 4: karonga m
ct = 0
for index, row in Kcorrlistm.iterrows():
    ax4.plot(lags,row, label=lbK+str(ct),linewidth=2)
    ct = ct+1
    
ax4.set_xlabel('Lag time (days)',size=8)
ax4.set_title('D.', size=8)
    
ax1.tick_params(axis='both',labelsize=8)
ax2.tick_params(axis='both',labelsize=8)
ax3.tick_params(axis='both',labelsize=8)
ax4.tick_params(axis='both',labelsize=8)

lines, labels = [(a + b) for a, b in zip(ax2.get_legend_handles_labels(), ax4.get_legend_handles_labels())]
f.legend(lines, labels, fancybox=False, shadow=False, ncol=1, frameon=False, loc='right', prop={'size': 8})
plt.show()
f.savefig('Figure_8.tiff',dpi=300)