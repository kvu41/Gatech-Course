#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:03:12 2019

@author: helenlu
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

nodes = pd.read_csv('nodes.txt', sep="\t", header=None)
edges = pd.read_csv('edges.txt', sep = "\t", header = None)

n = 1490

# build adjacency matrix
A = np.zeros((n,n))

for i in range(len(edges)):
    point1 = edges.iloc[i,0]
    point2 = edges.iloc[i,1]
    if A[point1-1,point2-1] == 0 or A[point2-1,point1-1] == 0:
        A[point1-1,point2-1] = 1
        A[point2-1,point1-1] = 1



D = np.zeros(n)
D = sum(A)

# count the number of isolated points
count = 0
for i in range(n):
    if D[i] != 0:
        count += 1

print(count)
D = np.diag(D)

# form special matrix L
L = D - A
# w is eigenvalues in ascending order
# v[:,i] is normalized eigenvector of corresponding eigenvalue w[i]
w,v = np.linalg.eigh(L)
# we want k = 2 eigenvectors of L corresponding to 2 SMALLEST eigenvalues
Z = v[:,:2]

# run k means on rows of Z
random.seed(200)

k = 2
min_val = min(Z[:,0])
max_val = max(Z[:,1])
#centroids[i] = [x,y]
centroids = {
        i: [random.uniform(min_val,max_val),random.uniform(min_val,max_val)]
        for i in range(k)}

fig = plt.figure(figsize=(5, 5))
plt.scatter(Z[:,0], Z[:,1], color='k')
colmap = {0: 'r', 1: 'g'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.show()

Z_df = pd.DataFrame(Z, columns=['x','y']) 

## Assignment Stage
def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

Z_df = assignment(Z_df, centroids)

fig = plt.figure(figsize=(5, 5))
plt.scatter(Z_df['x'], Z_df['y'], color=Z_df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.show()

# update stage
old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(Z_df[Z_df['closest'] == i]['x'])
        centroids[i][1] = np.mean(Z_df[Z_df['closest'] == i]['y'])
    return k

centroids = update(centroids)
    
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(Z_df['x'], Z_df['y'], color=Z_df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=0.01, head_length=0.02, fc=colmap[i], ec=colmap[i])
plt.show()

## Repeat Assigment Stage

Z_df = assignment(Z_df, centroids)

# Plot results
fig = plt.figure(figsize=(5, 5))
plt.scatter(Z_df['x'], Z_df['y'], color=Z_df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.show()

# Continue until all assigned categories don't change any more
while True:
    closest_centroids = Z_df['closest'].copy(deep=True)
    centroids = update(centroids)
    Z_df = assignment(Z_df, centroids)
    if closest_centroids.equals(Z_df['closest']):
        break
    
# Plot results
fig = plt.figure(figsize=(5, 5))
plt.scatter(Z_df['x'], Z_df['y'], color=Z_df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.show()

acc_count = 0
nonnan_count = 0
nodes_isnull = nodes.loc[:,2].isnull()
for i in range(n):
    if not nodes_isnull.iloc[i]:
        nonnan_count += 1
        if Z_df['closest'][i] == int(nodes.iloc[i,2]):
            acc_count += 1
        
print("The clustering accuracy is: " + str(acc_count/nonnan_count))