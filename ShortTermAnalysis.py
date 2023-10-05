# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:47:23 2023

@author: hmtha
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

df = pd.read_csv(r"Results\S-Deficit11-[7]-25-0.csv")
# df = pd.read_csv(r"C:\Users\hmtha\Desktop\FIRM_Australia\Results\S-Deficit21-[10]-25-0.csv")
# df['idx'] = pd.to_datetime(df['Date & time'], format = '%a -%d %b %Y %H:%M')
# df['transmission'] = df.iloc[:,19:].sum(axis=1)
df['idx'] = range(len(df))
df['Date & time'] = pd.to_datetime(df['Date & time'], format='%a -%d %b %Y %H:%M')
df['Date & time'] = df['Date & time']

deficitMask = df['eventDeficit'] > 0

ncols = [col for col in df.columns if col != 'Date & time']

ndf = (df[ncols]/df[ncols].mean()).fillna(0)
ndf['idx'] = range(len(df))

ndf = ndf.iloc[624:, :]
df = df.iloc[624:, :]

fig, ax = plt.subplots(figsize=(8,5))

pcols=['PHES-Charge', 'Hydropower', 'Biomass', 'Solar photovoltaics', 'eventPower', 
        'PHES-power', 'Energy spillage']#, 'PHES-Charge']
colors=['red', 'blue','purple','yellow','green','cyan','grey']#, 'red']

# ax.stackplot(ndf['idx'], *(ndf[col] for col in pcols), labels = pcols)
ax.stackplot(df['Date & time'], *(df[col] for col in pcols), labels = pcols, colors = colors)
# ax.plot(df['idx'], df['PHES-power']+df['Wind']+df['Solar photovoltaics']+df['Hydropower']+df['Biomass']+df['PHES-Charge'])


ax.plot(df['Date & time'], df['Operational demand (original)'], color='black', linestyle='--')
ax.axvline(df.loc[deficitMask, 'Date & time'].values[-1], color='r', linewidth=0.5, alpha = 0.5)
ax.axvline(df.loc[deficitMask, 'Date & time'].values[-1]-np.timedelta64(30*103, 'm'),color='r', linewidth=0.5, alpha = 0.5)

# ax.plot(df['idx'], df['PHES-Storage'], linestyle='--', color='red', label='PHES')
# ax.plot(df['idx'], df['eventStorage'], linestyle=':', color='blue', label='S-PHES')

ax2 = ax.twinx()
ax2.plot(df['Date & time'], df['PHES-Storage']-df['eventStorage'], linestyle='--', color='r', label='storageDifference')
# ax2.plot(df['idx'], df['eventSpillage'], linestyle='--', color='r', label='eventSpillage')
# ax2.plot(df['idx'], df['eventStorage'], linestyle=':', color='b', label='eventStorage')

# ax.plot(df['Wind'], label ='normal')
# ax.plot(df['eventPower'], label='event')


# ax.set_xlim([None, 1000])
plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

ax.legend()
ax2.legend()
plt.show()


