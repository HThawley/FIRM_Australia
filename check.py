import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

x = pd.read_csv('Results/OpHist11-1.csv')
x = x.loc[x.iloc[:,0] < 250, :]

x.iloc[:,0].hist()
plt.show()