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


scenario = 21
eventZone = np.array([7])
n_year = 25
event = 'e'

df = pd.read_csv(f"Results/S-Deficit{scenario}-{eventZone}-{n_year}-0-{event}.csv")
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

ndf = ndf.iloc[720:, :]
df  =  df.iloc[720:, :]

df['expec%'] = 100*df['PHES-Storage']/df['PHES-Storage'].max()
df['HILP%'] = 100*df['eventStorage']/df['PHES-Storage'].max()

df['Hydro & Bio (GW)'] = df['Hydropower'] + df['Biomass']

edf = df[['Date & time', 'Hydro & Bio (GW)', 'Solar photovoltaics', 'eventPower', 
       'eventPHES-power', 'eventSpillage', 'eventStorage', 'HILP%']]

df = df.rename(columns={
    #, 'Hydropower':'Hydro & Bio'
    #, 'Biomass':'Hydro & Bio'
    'Wind':'Wind (GW)'
    , 'Solar photovoltaics':'Solar (GW)'
    , 'PHES-power':'Storage (GW)'
    , 'Energy spillage':'Spillage (GW)'
    })

edf = edf.rename(columns={
    #, 'Hydropower':'Hydro & Bio'
    #, 'Biomass':'Hydro & Bio'
    'Solar photovoltaics':'Solar (GW)'
    , 'eventPHES-power':'Storage (GW)'
    , 'eventSpillage':'Spillage (GW)'
    , 'eventPower':'Wind (GW)'
    })


df['0'], edf['0'] = 0,0
pcols=['Hydro & Bio (GW)', 'Solar (GW)', 'Wind (GW)', 'Storage (GW)', 'Spillage (GW)']
ecols=['Hydro & Bio (GW)', 'Solar (GW)', 'Wind (GW)', 'Storage (GW)', 'Spillage (GW)']

df[pcols+['Demand']] = df[pcols+['Demand']]/1000.
edf[ecols] = edf[ecols]/1000.

fig, axs = plt.subplots(2, 1, figsize=(11, 6))
plt.subplots_adjust(hspace=0.9)

cmap = sns.color_palette("flare", as_cmap=True)

ax1, ax2 = axs[0], axs[1]

l = ax1.stackplot(
    df['Date & time']
    , *(df[col] for col in pcols)
    , labels = pcols
    # , colors = cmap(np.rint(np.arange(len(pcols))*cmap.N/(len(pcols)-1)).astype(int))
    )

l[0].set_color(sns.color_palette()[5])
l[1].set_color(sns.color_palette()[0])
l[2].set_color(sns.color_palette()[1])
l[3].set_color(sns.color_palette()[2])
l[4].set_color(sns.color_palette()[4])

# fixes extra ring caused by plotting negative spillage
l = ax1.stackplot(
    df['Date & time']
    , *(df[col] for col in ['Hydro & Bio (GW)', 'Solar (GW)', 'Wind (GW)', 'Storage (GW)', '0'])
    , colors = [(1,1,1,0)]
    )
l[-1].set_color((1,1,1,1))

l = ax2.stackplot(
    edf['Date & time']
    , *(edf[col] for col in ecols)
    , labels = pcols #p not e, so the legend is good
    # , colors = cmap(np.rint(np.arange(len(pcols))*cmap.N/(len(pcols)-1)).astype(int))
    )

l[0].set_color(sns.color_palette()[5])
l[1].set_color(sns.color_palette()[0])
l[2].set_color(sns.color_palette()[1])
l[3].set_color(sns.color_palette()[2])
l[4].set_color(sns.color_palette()[4])

# fixes extra ring caused by plotting negative spillage
l = ax2.stackplot(
    edf['Date & time']
    , *(edf[col] for col in ['Hydro & Bio (GW)', 'Solar (GW)', 'Wind (GW)', 'Storage (GW)', '0'])
    , colors = [(1,1,1,0)]
    )
l[-1].set_color((1,1,1,1))

ax1.plot(
    df['Date & time']
    , df['Demand']
    , color = 'cyan'#sns.color_palette()[len(pcols)+1]
    , linestyle='-'
    , linewidth = 1.25
    , label = 'Demand (GW)'
    )

ax2.plot(
    df['Date & time']
    , df['Demand']
    , color = 'cyan'#sns.color_palette()[len(pcols)+1]
    , linestyle='-'
    , linewidth = 1.25
    , label = 'Demand (GW)'
    )

ax1.axvline(df.loc[deficitMask, 'Date & time'].values[-1], color='r', linewidth=0.5, alpha = 0.5)
ax1.axvline(df.loc[deficitMask, 'Date & time'].values[-1]-np.timedelta64(int(125.5*60), 'm'),color='r', linewidth=0.5, alpha = 0.5)
ax2.axvline(df.loc[deficitMask, 'Date & time'].values[-1], color='r', linewidth=0.5, alpha = 0.5)
ax2.axvline(df.loc[deficitMask, 'Date & time'].values[-1]-np.timedelta64(int(125.5*60), 'm'),color='r', linewidth=0.5, alpha = 0.5)

ax12 = ax1.twinx()
ax12.plot(
    df['Date & time']
    , df['expec%']
    , linestyle='--'
    , color='grey'
    , label='Regular Storage (%)'
    )
ax12.set_ylim(0,None)

ax22 = ax2.twinx()
ax22.plot(
    edf['Date & time']
    , edf['HILP%']
    , linestyle='--'
    , color='black'
    , label='HILP Storage (%)'
    )

ax22.plot(
    df['Date & time']
    , df['expec%']
    , linestyle='--'
    , color='grey'
    , label='Regular Storage (%)'
    )
# ax22.plot(
#     df['Date & time']
#     , df['PHES-Storage'] - df['eventStorage']
#     , linestyle='--'
#     , color='black'
#     , label='Storage Difference'
#     )

ax22.set_ylim(0,None)
from datetime import datetime as dt

ax1.set_xticks([dt(2025, 6, i, j, 0) for i in range(19, 25) for j in (12,)])
ax2.set_xticks([dt(2025, 6, i, j, 0) for i in range(19, 25) for j in (12,)])

ax1.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d %H:%M'))
ax2.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d %H:%M'))

plt.setp(ax1.get_xticklabels(), rotation=-45, ha='left')
plt.setp(ax2.get_xticklabels(), rotation=-45, ha='left')

ax1.set_title('a) Energy supply-demand under regular conditions')
ax2.set_title('b) Energy supply-demand under HILP conditions')

ax1.set_ylabel('Power (GW)')
ax2.set_ylabel('Power (GW)')

ax12.set_ylabel('Energy Storage (%)')
ax22.set_ylabel('Energy Storage (%)')

ax1.set_xlabel('Date and Time')
ax2.set_xlabel('Date and Time')

for axs in ((ax1, ax12), (ax2, ax22)):
    pos = axs[0].get_position()
    axs[0].set_position([pos.x0, pos.y0, pos.width*0.92, pos.height])
    
    lns, labs = axs[0].get_legend_handles_labels()
    lns2, labs2 = axs[1].get_legend_handles_labels()
    
    axs[0].legend(
        lns + lns2
        , labs + labs2
        , loc = 'center right'
        , bbox_to_anchor = (1.33, 0.5)
        )


# ax22.legend()
plt.show()


