# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:56:50 2021

@author: lcmok
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np
from PIL import Image
import io

#import data
C0 = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Scriptsdataforfigures/C0_fig.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
K2 = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Scriptsdataforfigures/K2_fig.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])
K0 = pd.read_csv('/Users/lcmok/OneDrive/Documenten/Lund University/THESIS/Publication/Scriptsdataforfigures/K0_fig.csv', delimiter = ';', header = 0, na_filter=True, index_col=[0])

lims = [150,350]

C0.index = pd.to_datetime(C0.index, dayfirst=True)
K0.index = pd.to_datetime(K0.index, dayfirst=True)
K2.index = pd.to_datetime(K2.index, dayfirst=True)

K0 = K0[K0.index.year<1991]
K2 = K2[K2.index.year<1991]

Kyears = np.asarray(K0.index.year)
Cyears = np.asarray(C0.index.year)
fig, ((sp1), (sp2), (sp3)) = plt.subplots(3, 1, sharex=False, sharey=True, dpi=300, figsize=(3.346, 3.346))

sp1.plot(C0, linewidth=2)
sp1.set_ylim(lims)
sp1.set_yticks(np.arange(lims[0], lims[1]+1, 100))
sp1.set_ylabel('Tb (K)', size=8)
sp1.tick_params(axis='both', which='major', labelsize=8)

sp2.plot(K0,linewidth=2)
sp2.set_ylim(lims)
sp2.set_yticks(np.arange(lims[0], lims[1]+1, 100))
sp2.set_ylabel('Tb (K)', size=8)
sp2.tick_params(axis='both', which='major', labelsize=8)

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

fig.savefig('Figure2.tiff',dpi=300)

#%%
# Save the image in memory in PNG format
png1 = io.BytesIO()
fig.savefig(png1, format="png")
# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
png2.save("Figure2.tiff")
png1.close()

#%%