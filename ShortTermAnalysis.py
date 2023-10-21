# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:47:23 2023

@author: hmtha
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as pltd
from datetime import timedelta, datetime
import seaborn as sns



df = pd.read_csv(r"Results\S-Deficit21-[7]-25-0-e.csv")
# df = pd.read_csv(r"C:\Users\hmtha\Desktop\FIRM_Australia\Results\S-Deficit21-[10]-25-0.csv")
# df['idx'] = pd.to_datetime(df['Date & time'], format = '%a -%d %b %Y %H:%M')
# df['transmission'] = df.iloc[:,19:].sum(axis=1)
df['idx'] = range(len(df))
df['Date & time'] = pd.to_datetime(df['Date & time'], format='%a -%d %b %Y %H:%M')
df['Date & time'] = df['Date & time']
df = df.rename(columns={'Operational demand (original)':'Demand'})

deficitMask = df['eventDeficit'] > 0

ncols = [col for col in df.columns if col != 'Date & time']

ndf = (df[ncols]/df[ncols].mean()).fillna(0)
ndf['idx'] = range(len(df))

ndf = ndf.iloc[624:, :]
df = df.iloc[624:, :]

pcols=['Hydropower', 'Biomass', 'Solar photovoltaics', 'Wind', 
        'PHES-power', 'Energy spillage']
ecols=['Hydropower', 'Biomass', 'Solar photovoltaics', 'eventPower', 
       'eventPHES-power', 'eventSpillage']


fig, axs = plt.subplots(2, 1, figsize=(8, 9), layout = 'constrained')

ax1, ax2 = axs[0], axs[1]

ax1.stackplot(
    df['Date & time']
    , *(df[col] for col in pcols)
    , labels = pcols
    , colors = [sns.color_palette()[i] for i in range(len(pcols))]
    )

ax2.stackplot(
    df['Date & time']
    , *(df[col] for col in ecols)
    , labels = pcols #p not e, so the legend is good
    , colors = [sns.color_palette()[i] for i in range(len(ecols))]
    )

ax1.plot(
    df['Date & time']
    , df['Demand']
    , color = sns.color_palette()[len(pcols)]
    , linestyle='--'
    )

ax2.plot(
    df['Date & time']
    , df['Demand']
    , color = sns.color_palette()[len(pcols)]
    , linestyle='--'
    )

ax1.axvline(df.loc[deficitMask, 'Date & time'].values[-1], color='r', linewidth=0.5, alpha = 0.5)
ax1.axvline(df.loc[deficitMask, 'Date & time'].values[-1]-np.timedelta64(126, 'h'),color='r', linewidth=0.5, alpha = 0.5)
ax2.axvline(df.loc[deficitMask, 'Date & time'].values[-1], color='r', linewidth=0.5, alpha = 0.5)
ax2.axvline(df.loc[deficitMask, 'Date & time'].values[-1]-np.timedelta64(126, 'h'),color='r', linewidth=0.5, alpha = 0.5)

ax12 = ax1.twinx()
ax12.plot(
    df['Date & time']
    , df['PHES-Storage']
    , linestyle='--'
    , color='grey'
    , label='Storage'
    )
ax12.set_ylim(0,None)

ax22 = ax2.twinx()
ax22.plot(
    df['Date & time']
    , df['eventStorage']
    , linestyle='--'
    , color='black'
    , label='HILP Storage'
    )

ax22.plot(
    df['Date & time']
    , df['PHES-Storage']
    , linestyle='--'
    , color='grey'
    , label='Regular Storage'
    )
# ax22.plot(
#     df['Date & time']
#     , df['PHES-Storage'] - df['eventStorage']
#     , linestyle='--'
#     , color='black'
#     , label='Storage Difference'
#     )

ax22.set_ylim(0,None)


ax1.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d %I:%M'))
ax2.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d %I:%M'))

plt.setp(ax1.get_xticklabels(), rotation=-45, ha='left')
plt.setp(ax2.get_xticklabels(), rotation=-45, ha='left')

ax1.set_title('a) Energy supply-demand under regular conditions')
ax2.set_title('b) Energy supply-demand under HILP conditions')

ax1.set_ylabel('Power (MW)')
ax2.set_ylabel('Power (MW)')

ax12.set_ylabel('Energy Storage (MWh)')
ax22.set_ylabel('Energy Storage (MWh)')

ax1.set_xlabel('Date and Time')
ax2.set_xlabel('Date and Time')

# ax22.legend()
plt.show()


