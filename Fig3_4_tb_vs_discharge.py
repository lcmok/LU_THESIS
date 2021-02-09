# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:37:41 2021

@author: lcmok
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np
from PIL import Image
import io

#Change path according to where your csv file is located (columns: D-M-YYYY date, discharge, cmc-ratio and m index. column headers over column 2,3, and 4: Q, rcmc and m).
C0 = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Scriptsdataforfigures/C0_fig_season.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
C0.index = pd.to_datetime(C0.index, dayfirst=True)
Cyears = np.asarray(C0.index.year)

#%%
C0 = C0[C0.index.year<2009]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

#Plot A
ax11 = ax1.twinx()
ax1.plot(C0.index, C0.Q, linewidth=1.5)
ax1.set_ylim([0,2500])
ax1.set_ylabel('Discharge ($m^3$ $s^{-1}$)')
ax1.set_title('A.')
ax11.plot(C0.index, C0.rcmc, 'k', linewidth=1.5)
ax11.set_ylim([0,0.25])
ax11.set_ylabel('rcmc', size=8)
#Plot discharge over sat signal
ax1.set_zorder(ax1.get_zorder()+1)
ax1.patch.set_visible(False)

#Plot B
ax22 = ax2.twinx()
ax2.plot(C0.index, C0.Q, linewidth=1.5)
ax2.set_ylim([0,2500])
ax2.set_title('B.')
ax22.plot(C0.index, C0.m, 'k', linewidth=1.5)
ax22.set_ylim([-4,6])
ax22.set_ylabel('m', size=8)
#Plot discharge over sat signal
ax2.set_zorder(ax2.get_zorder()+1)
ax2.patch.set_visible(False)

plt.tight_layout()

#%%
ax3.tick_params(axis='both', which='major', labelsize=8)
ax3.set_xlabel('Time (years)', size=8)
#%%
sp2=fig.subplot(2,2,2)
sp2.plot(K0,linewidth=2)
sp2.set_ylim(lims)
sp2.set_yticks(np.arange(lims[0], lims[1]+1, 100))
sp2.set_ylabel('Tb (K)', size=8)
sp2.tick_params(axis='both', which='major', labelsize=8)

sp3=fig.subplot(2,2,3)
sp3.plot(K2['Cd2'],linewidth=2, label='Cd')
sp3.plot(K2['Cw2'],linewidth=2, label='Cw')
sp3.plot(K2['M2'],linewidth=2, label='M')
sp3.set_ylim(lims)
sp3.set_yticks(np.arange(lims[0], lims[1]+1, 100))
sp3.set_ylabel('Tb (K)', size=8)
sp3.set_xlabel('Time (years)', size=8)
sp3.tick_params(axis='both', which='major', labelsize=8)

#Add legend
llines, llabels = sp3.get_legend_handles_labels()
plt.legend(llines, llabels, fancybox=False, shadow=False, ncol=3, bbox_to_anchor=(0.8,-0.7), frameon=False, prop={'size': 8})
plt.tight_layout()

sp1.set_title('A.', size=8, loc='left')
sp2.set_title('B.', size=8, loc='left')
sp3.set_title('C.', size=8, loc='left')
plt.show()

fig.savefig('Figure2.tiff',dpi=300) #this will save your file with 300 dpi
