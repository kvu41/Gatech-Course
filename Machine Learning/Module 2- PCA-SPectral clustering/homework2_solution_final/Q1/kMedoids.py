import random
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
random.seed(45673)

class kMedoids:
    def __init__(self,k, num_iterations = 10):
        self.k = k
        self.num_iterations = num_iterations
        self.clusters = defaultdict(list)
        self.medoids = None
        self.prior_cost = float("inf")

    
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


    def nearest_mediod(self, s, c):
        """
        nearest_mediod: Find the nearest medoid to each sample
        inputs: 
        s = a sample 
        c = array of centroids 

        outputs: the index of the centroid that is closest to the sample s
        """
        idx = None
        distance = float("inf")
        for i, centroid in enumerate(c):
            d = self.m_distance(s, centroid)
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
            centroid_i = self.nearest_mediod(s, c)
            clusters[centroid_i].append(idx)
        return clusters


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

    def predict(self, centroids, d):
        """
        predict: predict the labels of each input data point
        inputs: 
        d = the input data 

        outputs: an arrray of labels 
        """

        clusters = self.assign_clusters(centroids, d)

        labels = self.assign_labels(clusters, d)

        return labels

    def fit(self, data):
        """
        fit: runs the k-medoids algorithm and returns the labels and mediods
        inputs: 
        data = the input data 

        outputs: an arrray of labels and an array of medoids 
        """

        total_cost = 0


        while (True and self.num_iterations !=0):
            """
            The medoids are initialized randomly for faster preprocessing 

            if costs do not decrease the loop breaks and the results are provided
            

            outputs: an arrray of labels and an array of medoids 
            """
            if self.medoids is None or not self.medoids:
                self.medoids = random.sample(list(data), self.k)
            else:
                random.shuffle(self.medoids)
                for _ in range(0, int(self.k/2)):
                    self.medoids.pop()
                self.medoids += random.sample(list(data), int(self.k/2))

            clusters = defaultdict(list)

            for d in data:
                temp = []
                for i in range(0, len(self.medoids)):
                    medoid = self.medoids[i]
                    temp.append(self.m_distance(medoid, d))
                index = np.argmin(temp)
                clusters[index].append(d)

            for idx in range(self.k):
                cluster = clusters[idx]
                medoid = self.medoids[idx]
                for i in range(len(cluster)):
                    item = cluster[i]
                    total_cost += self.m_distance(medoid, item)

            if total_cost < self.prior_cost:
                self.prior_cost = total_cost
            else:
                break

            self.num_iterations -= 1
        





        labels = self.predict(self.medoids, data)

        return labels, self.medoids
