# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:31:02 2024

@author: u6942852
"""
import numpy as np 
# from Setup import ub, lb
# ub, lb = np.array(ub), np.array(lb)
ub = np.array([  50.,   50.,   50.,   50.,   50.,   50.,   50.,   50.,   50.,
         50.,   50.,   50.,   50.,   50.,   50.,   50., 5000.])
lb = np.array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       5.98443071, 0.        ])
#%%

from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import OPTICS, cluster_optics_dbscan, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

points = np.genfromtxt('Results/OpHist11.csv', delimiter = ',')

points = points[points[:,0] != np.inf, :]
# points = points[points[:,0] < 1.1*points[:,0].min(), :]

# points = points[:len(points)//4, :]

costs, points = points[:,0], points[:,1:]

points = (points-lb)/(ub-lb)
points = np.hstack((costs.reshape(-1, 1), points))

#%%
i_1=0
for i in (1.01,1.02,1.05,1.1,1.2,1.5):
    p = points[points[:,0] <= i*(points[:,0].min()), :]
    p = p[p[:,0] >= i_1*(p[:,0].min()), :]
    p[:,0] /= 100
    fig, ax = plt.subplots()
    sns.boxplot(p, ax=ax)
    ax.set_title(str(i))

    i_1 = 0

#%%      

def distance(centre, point):
    return sum((c-p)**2 for c, p in zip(centre,point))

kmax=10
avedistances = np.zeros(kmax-2)
sils = np.zeros(kmax-2)

for k in range(7, kmax):
    kmc = KMeans(n_clusters=k).fit(points)
    centroids = kmc.cluster_centers_
    labels = kmc.labels_
    aves= np.zeros(k)
    for i in range(len(points)):
        clus = labels[i]
        aves[clus] += distance(centroids[clus], points[i])**0.5
        
    avedistances[k-2] = np.mean(aves)
    sils[k-2] = silhouette_score(points, labels)
    print(k, end=' ')

fig, ax = plt.subplots()
ax.plot(np.arange(2,kmax), avedistances)
ax.set_xticks(range(kmax))
ax.set_ylabel("distance")
ax.set_xlabel("no. clusters")

fig, ax = plt.subplots()
ax.plot(np.arange(2,kmax), sils)
ax.set_xticks(range(kmax))
ax.set_ylabel("silhouette score")
ax.set_xlabel("no. clusters")

raise KeyboardInterrupt
#%%
kmc = KMeans(n_clusters = 5)
kmc.fit_transform(points)

data = pd.DataFrame(points)
# data['cost'] = costs
data['labels'] = kmc.labels_

dgby = data.groupby('labels')

d = {}
for name, group in dgby:
    g = group.describe().drop(index=['count'])
    d = {**d, **{name:g}}
    cost = g.loc['mean', 'cost']
    group['cost'] /= 100
    del group['labels']
    fig, ax = plt.subplots()
    sns.boxplot(group, ax=ax)
    ax.set_title(f"Group {name}, cost: {round(cost, 2)}")
    
    print(name,cost )
raise KeyboardInterrupt
#%%
# best_s = -1
# best={'xi':0,
#       'mcs':0,
#       'me':0}

# for xi in (0.001, 0.01, 0.05, 0.1, 0.2):
#     for mcs in (2,10,50,100):
#         for me in (0.001, 0.01, 0.1, 1, 10, np.inf):
#             clust = OPTICS(min_samples=2, xi=xi, min_cluster_size=mcs, max_eps=me, n_jobs=-1).fit(points)
#             print(xi, mcs, me)
#             if len(np.unique(clust.labels_)) > best_s:
#                 best['xi'], best['mcs'], best['me'] = xi, mcs, me
#                 best_s = len(np.unique(clust.labels_))
            
# clust = OPTICS(min_samples=2, xi=best['xi'], min_cluster_size=best['mcs'], max_eps=best['me'], n_jobs=-1).fit(points)
# raise KeyboardInterrupt

#%%
raise KeyboardInterrupt
clust = OPTICS(min_samples=5, xi=0.001, min_cluster_size=2, max_eps=1.0, n_jobs=-1).fit(points)

eps1, eps2 = 0.5, 2

labels_eps1 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=eps1,
)
labels_eps2 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=eps2,
)

space = np.arange(len(points))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in zip(range(0, 5), colors):
    pointsk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(pointsk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
ax1.set_ylabel("Reachability (epsilon distance)")
ax1.set_title("Reachability Plot")

# OPTICS
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in zip(range(0, 5), colors):
    pointsk = points[clust.labels_ == klass]
    ax2.plot(pointsk[:, 0], pointsk[:, 1], color, alpha=0.3)
ax2.plot(points[clust.labels_ == -1, 0], points[clust.labels_ == -1, 1], "k+", alpha=0.1)
ax2.set_title("Automatic Clustering\nOPTICS")

# DBSCAN at 0.5
colors = ["g.", "r.", "b.", "c."]
for klass, color in zip(range(0, 4), colors):
    pointsk = points[labels_eps1 == klass]
    ax3.plot(pointsk[:, 0], pointsk[:, 1], color, alpha=0.3)
ax3.plot(points[labels_eps1 == -1, 0], points[labels_eps1 == -1, 1], "k+", alpha=0.1)
ax3.set_title(f"Clustering at {eps1} epsilon cut\nDBSCAN")

# DBSCAN at 2.
colors = ["g.", "m.", "y.", "c."]
for klass, color in zip(range(0, 4), colors):
    pointsk = points[labels_eps2 == klass]
    ax4.plot(pointsk[:, 0], pointsk[:, 1], color, alpha=0.3)
ax4.plot(points[labels_eps2 == -1, 0], points[labels_eps2 == -1, 1], "k+", alpha=0.1)
ax4.set_title(f"Clustering at {eps2} epsilon cut\nDBSCAN")
#%%
raise KeyboardInterrupt
norm = Normalizer().fit_transform(points)

# pca = PCA().fit(points[:,:-1])
pca = PCA().fit(norm)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.yticks(np.arange(0, 110, 10)/100)
plt.grid(True)
plt.grid(True, which='both', axis='y')