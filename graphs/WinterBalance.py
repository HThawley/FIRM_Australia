# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:43:27 2023

@author: hmtha
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as pltd
from datetime import timedelta, datetime
import seaborn as sns
from math import ceil
import os 

scenario = 21
eventZone = np.array([7])
n_year = 25
event = 'e'

os.chdir(r'C:\Users\hmtha\Desktop\FIRM_Australia')

df = pd.read_csv(f"CostOptimisationResults/Sx/S{scenario}-None.csv")

df['idx'] = range(len(df))
df['Date & time'] = pd.to_datetime(df['Date & time'], format='%a -%d %b %Y %H:%M')
df['Date & time'] = df['Date & time']
df = df.rename(columns={'Operational demand (original)':'Demand'})

df['expec%'] = 100*df['PHES-Storage']/df['PHES-Storage'].max()

ncols = [col for col in df.columns if col != 'Date & time']

june=True
year = 2029
yeardif = year-2020
leaps = ceil(yeardif/4)
if june: leaps+=1

df = df.iloc[48*leaps+17520*yeardif+8832:48*(1+leaps)+17520*yeardif+8832+1, :]


ndf = (df[ncols]/df[ncols].mean()).fillna(0)
ndf['idx'] = range(len(df))

df['Hydro & Bio (GW)'] = df['Hydropower'] + df['Biomass']

edf = df[['Date & time', 'Hydro & Bio (GW)', 'Solar photovoltaics', 'eventPower', 
       'eventPHES-power', 'eventSpillage', 'eventStorage']]

df = df.rename(columns={
    #, 'Hydropower':'Hydro & Bio'
    #, 'Biomass':'Hydro & Bio'
    'Wind':'Wind (GW)'
    , 'Solar photovoltaics':'Solar (GW)'
    , 'PHES-power':'Storage (GW)'
    , 'Energy spillage':'Spillage (GW)'
    , 'PHES-Charge':'Storage Charge (GW)'
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


fig, ax2 = plt.subplots(figsize=(8, 6), dpi = 2200)

l = ax2.stackplot(
    edf['Date & time']
    , *(edf[col] for col in ecols)
    , labels = pcols #p not e, so the legend is good
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


ax2.plot(
    df['Date & time']
    , df['Demand']
    , color = 'cyan'#sns.color_palette()[len(pcols)+1]
    , linestyle='-'
    , linewidth = 1.25
    , label = 'Demand (GW)'
    )

ax22 = ax2.twinx()
ax22.plot(
    df['Date & time']
    , df['expec%']
    , linestyle='--'
    , color='grey'
    , label='Storage (%)'
    )

ax22.set_ylim(0,None)
from datetime import datetime as dt

# ax1.set_xticks([dt(2025, 6, i, j, 0) for i in range(19, 25) for j in (12,)])
# ax2.set_xticks([dt(2025, 6, i, j, 0) for i in range(19, 25) for j in (12,)])

# ax1.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d %H:%M'))
ax2.xaxis.set_major_formatter(pltd.DateFormatter('%H:%M'))

# plt.setp(ax1.get_xticklabels(), rotation=-45, ha='left')
plt.setp(ax2.get_xticklabels(), rotation=-45, ha='left')

# ax1.set_title('a) Energy supply-demand under regular conditions')
ax2.set_title('Typical daily supply-demand balance (Winter)')

# ax1.set_ylabel('Power (GW)')
ax2.set_ylabel('Power (GW)')

# ax12.set_ylabel('Energy Storage (%)')
ax22.set_ylabel('Energy Storage (%)')

# ax1.set_xlabel('Date and Time')
ax2.set_xlabel('Date and Time')

for axs in (#(ax1, ax12),
             (ax2, ax22),):
    pos = axs[0].get_position()
    axs[0].set_position([pos.x0, pos.y0, pos.width*0.92, pos.height])
    
    lns, labs = axs[0].get_legend_handles_labels()
    lns2, labs2 = axs[1].get_legend_handles_labels()
    
    axs[0].legend(
        lns + lns2
        , labs + labs2
        , loc = 'center right'
        , bbox_to_anchor = (1.43, 0.5)
        )


# ax22.legend()
plt.show()

#%%

'''
from datetime import datetime as dt

wind = pd.read_csv('Data/wind.csv', dtype = float)
wind['dt'] = wind[['Year', 'Month', 'Day', 'Interval']].apply(
    lambda x: dt(int(x[0]), int(x[1]), int(x[2]), int((x[3]-1)//2), int(30*((x[3]-1)%2))), axis = 1)

wind = wind.loc[(wind['dt']<df['Date & time'].max()) & (wind['dt']>df['Date & time'].min())]

fig, ax = plt.subplots(figsize = (8,4))

ax.plot(
        wind['dt']
        , wind['N_Southern NSW Tablelands']*100
        , color = sns.color_palette('dark')[1]
        , label = 'Southern NSW Tablelands'
        )
ax.plot(
        wind['dt']
        , wind['Q_Fitzroy']*100
        , color = sns.color_palette('dark')[0]
        , label = 'Fitzroy'
        )

ax.set_xticks([dt(2025, 6, i, j, 0) for i in range(19, 25) for j in (12,)])
ax.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d %H:%M'))
plt.setp(ax.get_xticklabels(), rotation=-45, ha='left')

ax.legend()
ax.set_title("Capacity factor over time")
ax.set_ylim(0,100)
ax.set_yticks(np.arange(6)*20)
ax.set_ylabel('Capacity Factor (%)')
ax.set_xlabel('Date and Time')

ax.axvline(df.loc[deficitMask, 'Date & time'].values[-1]-np.timedelta64(int(127.5*60), 'm'),color='r', linewidth=0.5, alpha = 0.5)
ax.axvline(df.loc[deficitMask, 'Date & time'].values[-1], color='r', linewidth=0.5, alpha = 0.5)


pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width*0.92, pos.height])

lns, labs = ax.get_legend_handles_labels()

ax.legend(
    lns 
    , labs
    , loc = 'center right'
    , bbox_to_anchor = (1.47, 0.5)
    )

plt.show()'''