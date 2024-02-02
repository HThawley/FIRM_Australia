# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:40:26 2023

@author: hmtha
"""

import pandas as pd
import matplotlib.pyplot as plt

scenario=21
zone = '[10]'
for state in ('QLD', 'TAS', 'NSW', 'SA', 'VIC'):
    df = pd.read_csv(fr'Results\S{scenario}{state}-{zone}-25-e.csv')
    # print(state, df[['Hydropower', 'Biomass', 'Solar photovoltaics', 'Wind', 'PHES-power']].sum().sum())
    print(state, df[['Energy spillage']].sum().sum())
    fig, ax = plt.subplots()
    df['Transmission'].hist()
    plt.show()

