# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 21:45:32 2021

@author: lcmok
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from PIL import Image
import io
import datetime
import matplotlib.dates as mdates
from sklearn.metrics import r2_score


def createmodel(xtrain, ytrain, xtest, ytest, degree):
    #Create model based on training data  
    model = np.polyfit(xtrain, ytrain, degree)
    poly1d = np.poly1d(model)
    
    #Plot regression line and training data
    x2 = np.linspace(np.min(xtrain), np.max(xtrain))
    y2 = poly1d(x2)
    #plt.figure()
    #ax2 = plt.subplot()
    #ax2.plot(xtrain, ytrain, "o", x2, y2)
    #ax2.set_title('Training data and regression line')
    
    #PREDICT VALUES
    predictions = poly1d(xtest)
    
    #Plot predictions against original test data
    #plt.figure()
    #ax3 = plt.subplot()
    #ax3.plot(xtest, ytest, "o", xtest, predictions, "o")
    #ax3.set_title('Predictions vs. test data')
    
    #r2
    rsq = r2_score(ytest, predictions)
    
    #spearman
    from scipy.stats import spearmanr
    coef, p = spearmanr(xtrain,ytrain)
    # interpret the significance
    alpha = 0.05
    if p > alpha:
     	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
     	print('Samples are correlated (reject H0) p=%.3f' % p)
    return predictions, rsq, coef, model, poly1d, x2, y2



#%%

#Enter location name here: karonga or chikwawa
location = 'karonga'

#ensure you have csvs saved on your computer with the following columns: index, sattelite signal, discharge (see main script for the code to get this)
#you will need a training set and a testing set of each index and each location you want to analyze
    
if location == 'chikwawa':
    cmctrain = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Data/C_rcmctrain.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
    cmctest = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Data/C_rcmctest.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
    mtrain = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Data/C_mtrain.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
    mtest = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Data/C_mtest.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
else:    
    cmctrain = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Data/K_rcmctrain.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
    cmctest = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Data/K_rcmctest.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
    mtrain = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Data/K_mtrain.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
    mtest = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Data/K_mtest.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
x = np.asarray(cmctrain['0'])
xt = np.asarray(cmctest['0'])
y = np.asarray(cmctrain['1'])
yt = np.asarray(cmctest['1'])
xm = np.asarray(mtrain['0'])
xmt = np.asarray(mtest['0']) 
y2 = np.asarray(mtrain['1'])
yt2 = np.asarray(mtest['1'])

# APPLY CORRELATION AND CREATE MODEL
predictioncmc, rsqcmc, spcmc, modelcmc, polycmc, cmcx2, cmcy2 = createmodel(x,y, xt, yt, 2)
print(rsqcmc)
print(spcmc)

# #Test normality
# pval, resids = normal_errors_assumption(modelcmc, xt, yt, predictioncmc, p_value_thresh=0.05)
# resids=resids.reset_index(drop=True)
# #Test homoscedasticity
# homoscedasticity_assumption(modelcmc, xt, yt, predictioncmc)

predictionm, rsqm, spm, modelm, polym, mx2, my2 = createmodel(xm, y2, xmt, yt2, 2)
print(rsqm)
print(spm)
        
# #Test normality
# pval, resids = normal_errors_assumption(modelm, xmt, yt, predictionm, p_value_thresh=0.05)
# resids=resids.reset_index(drop=True)
# #Test homoscedasticity
# homoscedasticity_assumption(modelm, xmt, yt, predictionm)

#%%
# FIGURE 5 and 6
#Plot regression line and training data

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=True, dpi=300, figsize=(3.346, 3.346))

ax1.scatter(x, y, s=1)
ax1.plot(cmcx2, cmcy2, 'c', linewidth=2)
ax1.set_xlabel('r$_{cmc}$', size=8)
ax1.set_ylabel('Discharge ($m^3$ $s^{-1}$)', size=8)
ax1.set_title('A.', size=8)
ax1.set_xticks([0, 0.10, 0.2])
ax1.tick_params(axis='both', which='major', labelsize=8)

ax2.scatter(xm, y2, s=1)
ax2.plot(mx2, my2, 'c', linewidth=2)
ax2.set_xlabel('m', size=8)
ax2.set_title('B.', size=8)
ax2.set_xticks([-5, 0, 5])
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='y', which='major', label1On=False)

#ax2.set_title('Karonga m (ρ = %.3f)' % spm)

if location == 'Chikwawa':
    lims = [500,1200]
    xoneone = np.linspace(500,1250,100)
    yoneone = np.linspace(500,1250,100)
    ticks = np.arange(500,1001)
else:
    lims = [0,100]
    xoneone = np.linspace(0,100,100)
    yoneone = np.linspace(0,100,100)
    #ticks = np.arange(0,1001)  

ax3.scatter(yt, predictioncmc, label='Data point', s=1)
ax3.plot(xoneone, yoneone, '--', color='r', label='1:1 line', linewidth=2)
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_xlabel('True values ($m^3$ $s^{-1}$)', size=8)
ax3.set_ylabel('Predictions ($m^3$ $s^{-1}$)', size=8)
ax3.set_title('C.', size=8)
ax3.tick_params(axis='both', which='major', labelsize=8)


# if location == 'mwakimeme':
#     ax1.set_title('Karonga rcmc (r2 = %.3f)' % rsqcmc)
# else:
#     ax1.set_title('Chikwawa rcmc (r2 = %.3f)' % rsqcmc)
#ax1.set_xticks(ticks)
#ax1.set_xticklabels(ticks)

ax4.scatter(yt2, predictionm, label='Data points', s=1)
ax4.plot(xoneone, yoneone, '--', color='r', label='1:1 line', linewidth=2)
ax4.set_xlim(lims)
ax4.set_ylim(lims)
ax4.set_xlabel('True values ($m^3$ $s^{-1}$)', size=8)
ax4.set_title('D.', size=8)
ax4.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout()
f.savefig('Figure6.tiff',dpi=300, bbox_inches='tight')
plt.show()
