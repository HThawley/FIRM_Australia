# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 00:17:40 2023

@author: hmtha
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 
import seaborn as sns



op = pd.read_csv(r"C:\Users\hmtha\Desktop\Optimisation_resultx21.csv")

co = op.iloc[1:,52:]
op = op.iloc[1:,:52]
co.columns = [col.split('.')[0] for col in co.columns]

op = pd.melt(
    op
    , id_vars = ['Scenario', 'Zone', 'n_year', 'trial'] 
    , value_vars = ['pv-N_Broken Hill (GW)',
           'pv-N_Central NSW Tablelands (GW)', 'pv-N_Central West NSW (GW)',
           'pv-N_Murray River (GW)', 'pv-N_North West New South Wales (GW)',
           'pv-N_Northern NSW Tablelands (GW)', 'pv-N_Riverland (GW)',
           'pv-Q_Darling Downs (GW)', 'pv-Q_Fitzroy (GW)',
           'pv-S_Eastern Eyre Peninsula (GW)', 'pv-S_Leigh Creek (GW)',
           'pv-S_Northern SA (GW)', 'pv-S_Riverland (GW)', 'pv-S_Roxby Downs (GW)',
           'pv-S_Western Eyre Peninsula (GW)', 'pv-V_Murray River (GW)',
           'wind-N_Broken Hill (GW)', 'wind-N_Central NSW Tablelands (GW)',
           'wind-N_Central West NSW (GW)', 'wind-N_Murray River (GW)',
           'wind-N_North West New South Wales (GW)',
           'wind-N_Northern NSW Tablelands (GW)', 'wind-N_Riverland (GW)',
           'wind-N_Southern NSW Tablelands (GW)', 'wind-Q_Darling Downs (GW)',
           'wind-Q_Fitzroy (GW)', 'wind-S_Eastern Eyre Peninsula (GW)',
           'wind-S_Leigh Creek (GW)', 'wind-S_Mid-North SA (GW)',
           'wind-S_Northern SA (GW)', 'wind-S_Riverland (GW)',
           'wind-S_South East SA (GW)', 'wind-S_Western Eyre Peninsula (GW)',
           'wind-S_Yorke Peninsula (GW)', 'wind-T_King Island (GW)',
           'wind-T_North East Tasmania (GW)', 'wind-T_North West Tasmania (GW)',
           'wind-T_Tasmania Midlands (GW)', 'wind-V_Gippsland (GW)',
           'wind-V_Moyne (GW)', 'wind-V_Murray River (GW)',
           'wind-V_Western Victoria (GW)', 'storage-NSW (GW)', 'storage-QLD (GW)',
           'storage-SA (GW)', 'storage-TAS (GW)', 'storage-VIC (GW)','storage (GWh)' ]
    , var_name = 'ZoneName'
    , value_name = 'capacity'
    )

fig, ax = plt.subplots()

sns.barplot(
    op.loc[op.loc[:, 'ZoneName'] != 'storage (GWh)', :]
    , x = 'ZoneName'
    , y = 'capacity'
    , hue = 'trial'
    )

