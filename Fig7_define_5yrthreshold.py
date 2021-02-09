# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:34:38 2020

@author: lonem
"""

#script to provide the return period of annual maxima at a certain location
#script based upon https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/floods-return-period-and-probability/
#check for additional documentation

#YOU HAVE THE FOLLOWING CHOICES FOR MODES:
#seasonally = looks at peaks within 12 months in Malawi, but shifts the 'end of the year' to April rather than December.
#yearly = looks at peaks within 12 months (January to December)
#onlyseason = looks at all values observed in the flood season (November to April) every year, so does not look at 12 months of data

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
import numpy as np
import datetime as dt

def calculate_return(df, colname):
    '''
    Add Documentation Here
    '''
    # Sort data smallest to largest
    sorted_data = df.sort_values(by=colname)
    # Count total obervations
    n = sorted_data.shape[0]
    # Add a numbered column 1 -> n to use in return calculation for rank
    sorted_data.insert(0, 'rank', range(1, 1 + n))
    # Calculate probability
    sorted_data["probability"] = (n - sorted_data["rank"] + 1) / (n + 1)
    # Calculate return
    sorted_data["return-years"] = (1 / sorted_data["probability"])
    return(sorted_data)

#%% IMPORT DATA
#PARAMETERS
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=False, dpi=300, figsize=(3.346, 3.346))

for location in ['chikwawa', 'mwakimeme']: #list of locations in analysis
    
    if location == 'chikwawa':
        #Add path to your file
        data = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Scriptsdataforfigures/C0_fig_season.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
    else:
        #Add path to your file
        data = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Scriptsdataforfigures/K2_fig_season.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
        
    data.index = pd.to_datetime(data.index,dayfirst=True)
    dis = data.Q
    cmcdf = data.rcmc
    Magf = data.m
    
    # CALCULATE RETURN PERIOD FOR KNOWN FLOOD EVENT USING ANNUAL MAXIMA (ONLY REALLY WORKS WITH LONG TERM DATA)
    #Add flood season column (to avoid non-detection of nov-dec floods if peaks occurred earlier in the year) e.g. The 2014 flood season covers May 2014 until April 2015
    
    #DISCHARGE
    diss = pd.DataFrame(dis.dropna())
    diss['year'] = diss.index.year
    diss["floodseason"] = diss["year"]
    diss.loc[diss.index.month < 5, "floodseason"] = diss["floodseason"]-1  #will make the year run from May to April instead of January to December
    diss.dropna()
    #RCMC
    cmcdfs = pd.DataFrame(cmcdf.dropna())
    cmcdfs["year"] = cmcdfs.index.year
    cmcdfs.columns = ['rcmc', 'year']
    cmcdfs["floodseason"] = cmcdfs["year"]
    cmcdfs.loc[cmcdfs.index.month < 5, "floodseason"] = cmcdfs["floodseason"]-1  #will make the year run from May to April instead of January to December
    cmcdfs.dropna()
    #M
    mags = pd.DataFrame(Magf.dropna())
    mags["year"] = mags.index.year
    mags.columns = ['m', 'year']
    mags["floodseason"] = mags["year"]
    mags.loc[mags.index.month < 5, "floodseason"] = mags["floodseason"]-1  #will make the year run from May to April instead of January to December
    mags.dropna()
    
    # CALCULATING MAXIMA:
    mode = input('Choose strategy. Type yearly, seasonally, daily or onlyseason, look at documentation above: ')
    #mode = 'seasonally' #uncheck if you want to pick a standard, then check the line above
                
    if mode == 'yearly':
        #CALCULATE ANNUAL MAXIMUM
        diss["date"] = diss.index
        disagg = diss.resample('AS').max()
        cmcdfs["date"] = cmcdfs.index
        cmcagg = cmcdfs.resample('AS').max()
        mags["date"] = mags.index
        magagg = mags.resample('AS').max()  
        maxornot = 'max'
        
    elif mode == 'seasonally':
        #CALCULATE MAXIMUM PER HYDROLOGICAL YEAR
        diss["date"] = diss.index
        # #resample based on flood season rather than year
        disagg = diss.groupby(by='floodseason').max()
        #add date column to preserve original moment of flooding
        cmcdfs["date"] = cmcdfs.index
        #resample based on flood season rather than year
        cmcagg = cmcdfs.groupby(by='floodseason').max()
        #add date column to preserve original moment of flooding
        mags["date"] = mags.index
        magagg = mags.groupby(by='floodseason').max()   
        maxornot = 'max'
    
    elif mode == 'daily':
        disagg = diss
        cmcagg = cmcdfs
        magagg = mags
    else: 
        print('error')
        exit()
        
    #Drop na, sort ascending
    disagg = disagg.dropna()
    disagg_sorted = calculate_return(disagg, 'Q')
    magagg = magagg.dropna()
    magagg_sorted = calculate_return(magagg, 'm')
    cmcagg = cmcagg.dropna()
    cmcagg_sorted = calculate_return(cmcagg, 'rcmc')
    
    if mode == 'daily':
        if maxornot == 'val' or 'quant':
            # Because these data are daily,
            # divide return period in days by 365 to get a return period in years
            disagg_sorted["return-years"] = disagg_sorted["return-years"] / 365
            magagg_sorted["return-years"] = magagg_sorted["return-years"] / 365
            cmcagg_sorted["return-years"] = cmcagg_sorted["return-years"] / 365
    
    #Create relationship return period vs. discharge
    y1 = disagg_sorted['Q']
    x1 = disagg_sorted['return-years']
    y2 = cmcagg_sorted['rcmc']
    x2 = cmcagg_sorted['return-years']
    y3 = magagg_sorted['m']
    x3 = magagg_sorted['return-years']

    pl = 10 #degree of polynomial you want to use
        
    if location == 'chikwawa':    
        #discharge
        z1 = np.polyfit(x1, y1, pl)
        p1 = np.poly1d(z1)
        ax1.plot(x1,p1(x1),"--", color='orange', linewidth=2)
        ax1.scatter(x1,y1, s=4)
        ax1.set_xticks(np.arange(0, 60, 20))
        ax1.set_yticks(np.arange(0,2500,500))
        ax1.set_title('A.', size=8)
        ax1.set_ylabel('Discharge ($m^3$ $s^{-1}$)', size=8)
        #rcmc
        z2 = np.polyfit(x2, y2, pl) #10 degree
        p2 = np.poly1d(z2)
        ax2.plot(x2,p2(x2),"--",color='orange', linewidth=2)
        ax2.scatter(x2,y2, s=4)
        ax2.set_xticks(np.arange(0, 60,20))
        ax2.set_yticks(np.arange(0.05,0.30,0.05))
        ax2.set_title('B.', size=8)
        ax2.set_ylabel('r$_{cmc}$', size=8)
        #m
        ax3.scatter(x3,y3, s=4)
        z3 = np.polyfit(x3, y3, pl) #10th degree
        p3 = np.poly1d(z3)
        ax3.plot(x3,p3(x3),"--",color='orange', linewidth=2)
        ax3.set_yticks(np.arange(0,6,1))
        ax3.set_xticks(np.arange(0, 60, 20))
        ax3.set_title('C.', size=8)
        ax3.set_ylabel('m', size=8)
    else:
        #discharge
        ax4.scatter(x1,y1, s=4)
        z4 = np.polyfit(x1, y1, 2)
        p4 = np.poly1d(z4)
        ax4.plot(x1,p4(x1),"--",color='orange', linewidth=2)
        ax4.set_xticks(np.arange(0, 60, 20))
        ax4.set_yticks(np.arange(0,600,100))
        ax4.set_title('D.', size=8)
        ax4.set_ylabel('Discharge ($m^3$ $s^{-1}$)', size=8)
        #rcmc
        ax5.scatter(x2,y2, s=4)
        z5 = np.polyfit(x2, y2, pl) #10 degree
        p5 = np.poly1d(z5)
        ax5.plot(x2,p5(x2),"--",color='orange', linewidth=2)
        ax5.set_xticks(np.arange(0, 60,20))
        ax5.set_yticks(np.arange(0.05,0.30,0.05))
        ax5.set_xlabel('Return period (years)', size=8)
        ax5.set_title('E.', size=8)
        ax5.set_ylabel('r$_{cmc}$', size=8)
        #m
        z6 = np.polyfit(x3, y3, pl) #10th degree
        p6 = np.poly1d(z6)
        ax6.plot(x3,p6(x3),"--",color='orange', linewidth=2)
        ax6.scatter(x3,y3, s=4)
        ax6.set_yticks(np.arange(0,6,1))
        ax6.set_xticks(np.arange(0, 60, 20))
        ax6.set_title('F', size=8)
        ax6.set_ylabel('m', size=8)

ax1.tick_params(axis='both',labelsize=8)
ax2.tick_params(axis='both',labelsize=8)
ax3.tick_params(axis='both',labelsize=8)
ax4.tick_params(axis='both',labelsize=8)
ax5.tick_params(axis='both',labelsize=8)
ax6.tick_params(axis='both',labelsize=8)
f.tight_layout(w_pad=-0.2)
#f.savefig('Figure_7.tiff')
plt.show()

#%% CALCULATE YOUR THRESHOLDS
rp = input("Enter the return years you want to calculate the discharge for:" )
#d5 = np.interp(x1,p1, rp)
#cmc5 = np.interp(x2,p2, rp2)
#mag5 = np.interp(x3,p3, rp3)

d5 = np.polyval(p1, int(rp))
cmc5 = np.polyval(p2, int(rp))
mag5 = np.polyval(p3, int(rp))
d52 = np.polyval(p4, int(rp))
cmc52 = np.polyval(p5, int(rp))
mag52 = np.polyval(p6, int(rp))

print('The discharge C is...')
print(d5)
print('The cmc C is...')
print(cmc5)
print('The m C is...')
print(mag5)
print('The discharge K is...')
print(d52)
print('The cmc K is...')
print(cmc52)
print('The m K is...')
print(mag52)

#%% IF YOU WANT TO FIND THE RP CORRESPONDING TO A CERTAIN DISCHARGE...

Q = input("Enter the discharge you want to calculate the rp for:" )
Q5 = p4(float(Q))

print(Q5)