import pandas as pd 
import csv
from Setup import scenario, args, F

x = pd.read_csv('Results/OpHist{}-{}.csv'.format(scenario, args.x), header=None)

x = x.iloc[x.iloc[:,0].idxmin(), :]
print(x[0], x[1:], F(x[1:].to_numpy()))
# with open('Results/Optimisation_resultx{}.csv'.format(scenario, args.x), 'w', newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(x.values)
