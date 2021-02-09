# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:01:33 2021

@author: lcmok
"""

#STEP 1: DOWNLOAD WGET AND ENSURE IT WORKS

#With this script you can download the Tb .nc files from the nsidc website. It was adapted from WNeisinghs version: https://github.com/hcwinsemius/satellite-cookbook/tree/master/NSIDC-AMSRE
#It makes use of the software Wget, which I installed on windows from here: https://eternallybored.org/misc/wget/
#Please take care to download the newest version, as the older versions do not work with the nsidc website.
#In order to make it work, change the path in the cmd (opdrachtprompt) to the folder where wget.exe is located, or copy the Wget files to your System32 folder (be careful not to touch anyuthing else)

#If you want to merge different platforms, please download in chunks (i.e. first your dmsp dataset, then the aqua, etc.). That way the suffixes are correct.
#This script was written for Windows, if it doesnt work because it cannot find the directories, the // might be the wrong way because you are using a different OS.

#PACKAGES AND LIBRARIES
import os
import scipy
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr #installed
import pyproj #installed
import datetime
import subprocess
import shutil
import numpy as np
from ipywidgets import interact
import pandas as pd
import pickle
import time
import nsidc2

#%% STEP 2: CHOOSE THE DATA YOU WANT AND TEST THE LINK

#MANUALLY CHANGE
sat = 'AQUA' #OPTIONS: NIMBUS, AQUA, DMSP

#Set start and end date for time period you want to download. Check suffix & resolution of data in question.
#start_date = datetime.datetime(2002, 6, 1) # Start of the AMSR-E period
#end_date = datetime.datetime(2011, 10, 4)  # use this end date to cover the whole AMSR-E period.
start_date = datetime.datetime(2011, 1, 1)
end_date = datetime.datetime(2011, 1, 5)

#STATIC VARIABLES (do not change)
if sat == 'NIMBUS' or sat == 'DMSP':
    freq = '37' #37 ghz
elif sat == 'AQUA':
    freq = '36' #36.5 ghz
    
res = '25'  # can also be 3.125,  12.5 or 25, although this depends on the chosen frequency as well
HV = 'H' # Horizontal or Vertical polarisation
AD = 'D' # Ascending or descending, descending contains imagery during the day, probably showing more contrast

#Rough bounding box around your poi (bottom left, top right)
#latitude, longitude

#MALAWI
#bounds = [(-18.,  30.55),
#          ( -8.3, 37.),]
#MALI
bounds = [(12.688, -12.583),
         (15.035612,  -9.361141)]

url = nsidc2.make_measures_url(start_date, res, freq, HV, AD, sat) #Uncheck this line to obtain the download url. This can be used if th combination of parameters is not coresponding to a download file

print(url) 
#click this printed url and see if it works. if it doesnt work, you can check if the satellite covers the date you are interested in or not, or if the resolution you are interested in is missing, for example
#%% STEP 3: DOWNLOAD THE DATA

#running this section may give problems if Wget is not operating properly on your computer, if the credential file/cookie file is missing on your pc
#you can test wget by going into nsidc2 and printing out the variable download_string in the function download_measures. If you copy this string into your cmd line and press enter, does it give an error as well?

try:
    credential_fn = os.path.abspath('logins.txt') #ensure this txt file actually ecists
    username, password = str(np.loadtxt(credential_fn, dtype=np.str)).split(',')
except:
    print('No credential file found, please put a txt file with <user>,<pass> in {:s}'.format(credential_fn))

url, success = nsidc2.download_measures(freq, res, HV, AD, start_date, username, password, sat) #To check url and success of download -> 0 = succes
fn_out_prefix = 'AMSRE'

#download command
nsidc2.download_measures_ts(freq, res, HV, AD, start_date, end_date, bounds, fn_out_prefix, username, password, sat)

print('----------------------------')
if success==0:
    print('SUCCESSFULL DOWNLOAD')
    print('----------------------------')
    print('SETTINGS:'' Frequency:',freq,'GHz.', 
      ' Resolution:',res, 'km.',
      ' Polarization:', HV,'.' 
      ' Ascending/descending:', AD)