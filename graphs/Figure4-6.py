# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:47:23 2023

@author: hmtha
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as pltd
import seaborn as sns
from datetime import datetime as dt


from graphutils import directory_up, adjust_legend

scenario = 15
eventZone = np.array([23])
# zoneName = 'S-NSW Tablelands'
zoneName = ''
n_year = 25 
event = 'e'
dpi = 200
deficit_index=0 #rank (0-indexed)


directory_up()

df = pd.read_csv(f"Results/SDeficit{deficit_index}-{scenario}-{eventZone}-{n_year}-{event}.csv")
df = df.iloc[-350:, :]
deficitMask = df['eventDeficit'] > 0


df['Date & time'] = pd.to_datetime(df['Date & time'], format='%a -%d %b %Y %H:%M')

df['expec%'] = 100*df['PHES-Storage']/df['PHES-Storage'].max()
df['HILP%'] = 100*df['eventStorage']/df['PHES-Storage'].max()

df['Hydro & Bio (GW)'] = df['Hydropower'] + df['Biomass']

df = df.rename(columns={
    'Wind':'Wind (GW)',
    'Solar photovoltaics':'Solar (GW)',
    'PHES-power':'Storage (GW)',
    'Energy spillage':'Spillage (GW)',
    'PHES-Charge':'Charging (GW)',
    'Operational demand (original)':'Demand (GW)',
    })

edf = df[['Date & time', 'eventH&B', 'Solar (GW)', 'eventWind', 
       'eventPHES-power', 'eventSpillage', 'expec%', 'HILP%', 'Demand (GW)', 'eventPHES-Charge']]

edf = edf.rename(columns={
    'Solar photovoltaics':'Solar (GW)',
    'eventPHES-power':'Storage (GW)',
    'eventSpillage':'Spillage (GW)',
    'eventWind':'Wind (GW)',
    'eventPHES-Charge':'Charging (GW)',
    'eventH&B':'Hydro & Bio (GW)',
    })


def graphWrapper():
    fig, axs = plt.subplots(2, 1, figsize=(8, 9), dpi=dpi)
    plt.subplots_adjust(hspace=0.52)
    ax1, ax2 = axs[0], axs[1]
    
    plot_axis(ax1, df, False)
    plot_axis(ax2, edf, True)
    
    ys = list(zip(ax1.get_ylim(), ax2.get_ylim()))

    ax1.set_ylim(min(ys[0]), max(ys[1]))
    ax2.set_ylim(min(ys[0]), max(ys[1]))
    
    plt.show()

def plot_axis(ax, df, event):
    
    cols=['Hydro & Bio (GW)', 'Solar (GW)', 'Wind (GW)', 'Storage (GW)']
    negs=['Charging (GW)', 'Spillage (GW)']
    df[cols+negs+['Demand (GW)']] = df[cols+ negs+['Demand (GW)']]/1000. # MW to GW
    
    # plot generation areas
    ax.stackplot(
        df['Date & time'],
        *(df[col] for col in cols), 
        labels = cols,
        colors = [sns.color_palette()[i] for i in (5,0,1,2)],
        )
  
    # plot negative values
    ax.stackplot(
        df['Date & time'],
        *(df[col] for col in negs),
        labels=negs,
        colors=[sns.color_palette()[3], sns.color_palette()[4]]
        )
    
    # plot demand line
    ax.plot(
        df['Date & time'],
        df['Demand (GW)'],
        color = 'cyan',
        linestyle='-',
        linewidth = 1.25,
        label = 'Demand (GW)',
        )
        
    # Plot battery levels on secondary axis
    ax1 = ax.twinx()
    ax1.plot(
        df['Date & time']
        , df['expec%']
        , linestyle='--'
        , color='grey'
        , label='Regular Storage (%)'
        )
    
    if event is True: 
        ax1.plot(
            df['Date & time']
            , df['HILP%']
            , linestyle='--'
            , color='black'
            , label='HILP Storage (%)'
            )
    
    #plot event markers
    #TO DO - more robust way based on .iloc
    ax.axvline(df.iloc[-96, :].loc['Date & time'], color='r', linewidth=0.5, alpha = 0.5)
    ax.axvline(df.iloc[-96, :].loc['Date & time']-np.timedelta64(int(127.5*60), 'm'),color='r', linewidth=0.5, alpha = 0.5)
    
    ax1.set_ylim(0,None)

    if event is True: 
        ax.set_title('b) Energy supply-demand under HILP conditions in '+zoneName)
    else:
        ax.set_title('a) Energy supply-demand under regular conditions in '+zoneName)

    # ax.set_xticks([dt(2025, 6, i, j, 0) for i in range(19, 25) for j in (12,)])
    ax.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.set_xlabel('Date and Time')
    ax.set_ylabel('Power (GW)')
    ax1.set_ylabel('Energy Storage (%)')

    adjust_legend([ax, ax1], 1.47, 0.5, 0.92)

graphWrapper()


#%%
raise KeyboardInterrupt

wind = pd.read_csv('Data/wind.csv', dtype = float)
wind['dt'] = wind.loc[:,['Year', 'Month', 'Day', 'Interval']].apply(
    lambda x: dt(int(x[0]), int(x[1]), int(x[2]), int((x[3]-1)//2), int(30*((x[3]-1)%2))), axis = 1)

wind = wind.loc[(wind['dt']<df['Date & time'].max()) & (wind['dt']>df['Date & time'].min())]


fig, ax = plt.subplots(figsize = (8,4), dpi=dpi)

ax.plot(
        wind['dt'],
        wind['N_Southern NSW Tablelands']*100,
        color = sns.color_palette('dark')[1],
        label = 'Southern NSW\nTablelands',
        )
ax.plot(
        wind['dt'],
        wind['Q_Fitzroy']*100,
        color = sns.color_palette('dark')[0],
        label = 'Fitzroy',
        )

# ax.set_xticks([dt(2025, 6, i, j, 0) for i in range(19, 25) for j in (12,)])
ax.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d %H:%M'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

ax.legend()
ax.set_title("Capacity factor over time")
ax.set_ylim(0,100)
ax.set_yticks(np.arange(6)*20)
ax.set_ylabel('Capacity Factor (%)')
ax.set_xlabel('Date and Time')

ax.axvline(df.iloc[-96, :].loc['Date & time'], color='r', linewidth=0.5, alpha = 0.5)
ax.axvline(df.iloc[-96, :].loc['Date & time']-np.timedelta64(int(127.5*60), 'm'),color='r', linewidth=0.5, alpha = 0.5)

adjust_legend(ax, 1.32, 0.5, 0.92)

plt.show()

