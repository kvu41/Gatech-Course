

import numpy as np
import pandas as pd



class KMeans():
    """
    Initializing the KMeans class which terminates at 100 iterations of the 
    algorithm

    k = num of cluster centers 
    """
    def __init__(self, k=2, num_iterations=100):
        self.centroids = []
        self.k = k
        self.num_iterations = num_iterations
        
    def m_distance(self, x, y):
        """
        m_distance: calculate the manhattan distance between two points 

        inputs: x and y vectors 

        outputs: euclidean distance between x and y

        """
        if(len(x) != len(y)):
            raise Exception("x and y are of unequal length")

        d = 0
        for i in range(len(x)):
            d += np.linalg.norm(x[i] - y[i])

        return d


    def e_distance(self,x, y):
        """
        e_distance: calculate the euclidean distance between two points 

        inputs: x and y vectors 

        outputs: euclidean distance between x and y

        """
        if(len(x) != len(y)):
            raise Exception("x and y are of unequal length")

        d = 0
        for i in range(len(x)):
            d += pow((x[i] - y[i]), 2)

        d = np.sqrt(d)
        return d



    def initialize_centroids(self, d):
        """
        initialize_centroids: Initialize the centroids as random points from 
        within the d. An alternative method could be the kmeans++ 
        initialization which intializes the centroids with the maximum distance 
        apart and sometimes yeilds better results. 

        inputs: d matrix or array

        outputs: array of intial centroids

        """
        s, f = np.shape(d)
        centroids = np.zeros((self.k, f))
        for i in range(self.k):
            c = d[np.random.choice(range(s))]
            centroids[i] = c
        return centroids

    

    def nearest_centroid(self, s, c):
        """
        nearest_centroid: Find the nearest centroid to each sample
        inputs: 
        s = a sample 
        c = array of centroids 

        outputs: the index of the centroid that is closest to the sample s
        """
        idx = None
        distance = float("inf")
        for i, centroid in enumerate(c):
            d = self.e_distance(s, centroid)
            if d < distance:
                idx = i
                distance = d
        return idx




    def assign_clusters(self, c, d):
        """
        assign_clusters: Assign each d point d to a centroid generating a 
        cluster 

        inputs: 
        d = the input d 
        c = the centroids c

        outputs: A list of clusters where each cluster is a list. 
        """
        clusters = [[] for _ in range(self.k)]
        for idx, s in enumerate(d):		
            centroid_i = self.nearest_centroid(s, c)
            clusters[centroid_i].append(idx)
        return clusters

    
    def find_centers(self, clstrs, d):
        """
        find_centers: Find the cluster center of the cluster, this step is done 
        after the cluster centers are assigned. 

        inputs: 
        d = the input d 
        clstrs = an array of clusters

        outputs: A list of clusters where each cluster is a list. 
        """
        f = np.shape(d)[1]
        centroids = np.zeros((self.k, f))
        for i, clstr in enumerate(clstrs):
            centroid = np.mean(d[clstr], axis=0)
            centroids[i] = centroid
        return centroids





    def assign_labels(self, clstrs, d):
        """
        assign_labels: label the the input d based on the clustering results

        inputs: 
        d = the input d 
        clsters = a array of clusters 

        outputs: the labels for the input d
        """
        y_pred = np.zeros(np.shape(d)[0])
        for idx, clstr in enumerate(clstrs):
            for sample in clstr:
                y_pred[sample] = idx
        return y_pred


    def fit(self, d):
        """
        fit: The primary function that performs the kmeans clustering to fit 
        the training data. 
        inputs: 
        d = the input data 

        outputs: the centroids after the algorithm has run for a certain number
        of iterations
        """

        centroids = self.initialize_centroids(d)


        for _ in range(self.num_iterations):

            clusters = self.assign_clusters(centroids, d)

            prev_centroids = centroids

            centroids = self.find_centers(clusters, d)

            # Stop condition if centroids are not changing
            diff = centroids - prev_centroids
            if not diff.any():
                break

        self.centroids = centroids
        return centroids

    
    def predict(self, d):
        """
        predict: predict the labels of each input data point
        inputs: 
        d = the input data 

        outputs: an arrray of labels 
        """


        if not self.centroids.any():
            raise Exception("""The training data has not been clustered. Call 
        the fit function first.""")

        clusters = self.assign_clusters(self.centroids, d)

        labels = self.assign_labels(clusters, d)

        return labels
