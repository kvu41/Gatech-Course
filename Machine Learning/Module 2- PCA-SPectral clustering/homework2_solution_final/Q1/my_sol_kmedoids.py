#% Your goal of this assignment is implementing your own K-medoids.
#% Please refer to the instructions carefully, and we encourage you to
#% consult with other resources about this algorithm on the web.
#%
#% Input:
#%     pixels: data set. Each row contains one data point. For image
#%     dataset, it contains 3 columns, each column corresponding to Red,
#%     Green, and Blue component.
#%
#%     K: the number of desired clusters. Too high value of K may result in
#%     empty cluster error. Then, you need to reduce it.
#%
#% Output:
#%     class: the class assignment of each data point in pixels. The
#%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
#%     of class should be either 1, 2, 3, 4, or 5. The output should be a
#%     column vector with size(pixels, 1) elements.
#%
#%     centroid: the location of K centroids in your result. With images,
#%     each centroid corresponds to the representative color of each
#%     cluster. The output should be a matrix with K rows and
#%     3 columns. The range of values should be [0, 255].
#%     
#%
#% You may run the following line, then you can see what should be done.
#% For submission, you need to code your own implementation without using
#% the kmeans matlab function directly. That is, you need to comment it out.

from kMedoids import kMedoids
import numpy as np

# def my_kmedoids(image_data, K):
#     kmedoids = kMedoids(n_clusters=K)
#     kmedoids.fit(image_data)
#     label = kmedoids.predict(image_data)
#     centroid = kmedoids.medoids.keys()
#     return label, centroid


# import random
# import numpy as np
# from collections import defaultdict
# def kMedoids(data, k, prev_cost=float("inf"), count=0, clusters=None, medoids=None):

#     cluster_sum = 0
#     random.seed(0)

#     while (True and count <= 10):

#         if medoids is None or not medoids:
#             medoids = random.sample(list(data), k)
#         else:
#             random.shuffle(medoids)
#             for _ in range(0, int(k/2)):
#                 medoids.pop()
#             medoids += random.sample(list(data), int(k/2))

#         clusters = defaultdict(list)

#         for item in data:
#             temp = []
#             for i in range(0, len(medoids)):
#                 med = medoids[i]
#                 if 1==2:
#                     break
#                 else:
#                     temp.append(np.linalg.norm(
#                         med[0]-item[0])+np.linalg.norm(med[1]-item[1]))
#             min_index = np.argmin(temp)
#             clusters[min_index].append(item)

#         for i in range(0, len(medoids)):
#             inter_cluster = clusters[i]
#             for j in range(0, len(inter_cluster)):
#                 item_cluster = inter_cluster[j]
#                 medoid = medoids[i]
#                 cluster_sum += (np.linalg.norm(medoid[0]-item_cluster[0]) +
#                                 np.linalg.norm(medoid[1]-item_cluster[1]))

#         if cluster_sum < prev_cost:
#             prev_cost = cluster_sum
#         else:
#             break

#         count += 1
    
#     def e_distance(x, y):
#         """
#         e_distance: calculate the euclidean distance between two points 

#         inputs: x and y vectors 

#         outputs: euclidean distance between x and y

#         """
#         if(len(x) != len(y)):
#             raise Exception("x and y are of unequal length")

#         d = 0
#         for i in range(len(x)):
#             d += pow((x[i] - y[i]), 2)

#         d = np.sqrt(d)
#         return d

#     def nearest_centroid(s, c):
#         """
#         nearest_centroid: Find the nearest centroid to each sample
#         inputs: 
#         s = a sample 
#         c = array of centroids 

#         outputs: the index of the centroid that is closest to the sample s
#         """
#         idx = None
#         distance = float("inf")
#         for i, centroid in enumerate(c):
#             d = e_distance(s, centroid)
#             if d < distance:
#                 idx = i
#                 distance = d
#         return idx

#     def assign_clusters(c, d):
#         """
#         assign_clusters: Assign each d point d to a centroid generating a 
#         cluster 

#         inputs: 
#         d = the input d 
#         c = the centroids c

#         outputs: A list of clusters where each cluster is a list. 
#         """
#         clusters = [[] for _ in range(k)]
#         for idx, s in enumerate(d):		
#             centroid_i = nearest_centroid(s, c)
#             clusters[centroid_i].append(idx)
#         return clusters


#     def assign_labels(clstrs, d):
#         """
#         assign_labels: label the the input d based on the clustering results

#         inputs: 
#         d = the input d 
#         clsters = a array of clusters 

#         outputs: the labels for the input d
#         """
#         y_pred = np.zeros(np.shape(d)[0])
#         for idx, clstr in enumerate(clstrs):
#             for sample in clstr:
#                 y_pred[sample] = idx
#         return y_pred

#     def predict(centroids, d):
#         """
#         predict: predict the labels of each input data point
#         inputs: 
#         d = the input data 

#         outputs: an arrray of labels 
#         """

#         clusters = assign_clusters(centroids, d)

#         labels = assign_labels(clusters, d)

#         return labels


#     labels = predict(medoids, data)
#     return labels, medoids

def my_kmedoids(image_data, K):
    clf = kMedoids(k=K)
    label, centroids = clf.fit(image_data)
    centroids = [list(centroid) for centroid in centroids]
    label = label.astype(int)
    return label, np.array(centroids)