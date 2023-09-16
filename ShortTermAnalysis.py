# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:47:23 2023

@author: hmtha
"""

import pandas as pd 
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\hmtha\Desktop\FIRM_Australia\Results\S-Deficit21-[18 26 28]-25-0.csv")
# df['idx'] = pd.to_datetime(df['Date & time'], format = '%a -%d %b %Y %H:%M')
# df['transmission'] = df.iloc[:,19:].sum(axis=1)
df['idx'] = range(len(df))

deficitMask = df['StormDeficit'] > 0

ncols = [col for col in df.columns if col != 'Date & time']

ndf = (df[ncols]/df[ncols].mean()).fillna(0)
ndf['idx'] = range(len(df))

ndf = ndf.iloc[624:, :]
df = df.iloc[624:, :]

fig, ax = plt.subplots()

pcols=['PHES-Charge', 'Hydropower', 'Biomass', 'Solar photovoltaics', 'StormPower', 
        'PHES-power', 'Energy spillage']#, 'PHES-Charge']
colors=['red', 'blue','purple','yellow','green','cyan','grey']#, 'red']

# ax.stackplot(ndf['idx'], *(ndf[col] for col in pcols), labels = pcols)
ax.stackplot(df['idx'], *(df[col] for col in pcols), labels = pcols, colors = colors)
# ax.plot(df['idx'], df['PHES-power']+df['Wind']+df['Solar photovoltaics']+df['Hydropower']+df['Biomass']+df['PHES-Charge'])


ax.plot(df['idx'], df['Operational demand (original)'], color='black', linestyle='--')
ax.axvline(ndf.loc[deficitMask, 'idx'].values, color='r', linewidth=0.5, alpha = 0.5)
ax.axvline(ndf.loc[deficitMask, 'idx'].values-103,color='r', linewidth=0.5, alpha = 0.5)

ax2 = ax.twinx()
ax2.plot(df['idx'], df['PHES-Storage']-df['StormStorage'], linestyle='--', color='b', label='storage')
# ax2.plot(df['idx'], df['StormStorage'], linestyle=':', color='b', label='stormStorage')


plt.legend()
plt.show()


