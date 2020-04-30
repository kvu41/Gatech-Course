import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random as rd
import math
from sklearn.cluster import KMeans

data = np.genfromtxt('data.dat')
label = np.genfromtxt('label.dat')
x = data.T

N = 1990
C = 2
P = 784
r = 60 # you can change the rank to see the influence

# define initial values
mean = np.zeros((P,2))
cov1 = np.identity(P)
cov2 = np.identity(P)
rand = rd.random()
pi = np.array([rand, 1-rand])
tau = np.zeros((N,C))
ll = []

### gaussian probability with low-rank approximation
def gaussian(x, pi, mean, cov1, cov2):
    u1,s1,v1=sp.linalg.svd(cov1,full_matrices=False)
    u1 = u1[:,:r]
    s1 = np.reshape(s1[:r],(r,1))
    u2,s2,v2=sp.linalg.svd(cov2,full_matrices=False)
    u2 = u2[:,:r]
    s2 = np.reshape(s2[:r],(r,1))
    new_x1 = u1.T @ x.T
    new_x2 = u2.T @ x.T
    new_mean1 = np.reshape(u1.T @ mean[:,0], (r,1))
    new_mean2 = np.reshape(u2.T @ mean[:,1], (r,1))
    temp1 = np.sum((new_x1-new_mean1)**2 / s1,axis=0)
    temp2 = np.sum((new_x2-new_mean2)**2 / s2,axis=0)
    gau1 = 1/np.sqrt(np.prod(s1)) *np.exp(-0.5*temp1)
    gau2 = 1/np.sqrt(np.prod(s2)) *np.exp(-0.5*temp2)
    return np.vstack((gau1, gau2)).T



# 2a
for i in range(N):
    if label[i] == 2:
        index2 = i
        break
for i in range(N):
    if label[i] == 6:
        index6 = i
        break
random_data2 = data[:,index2]
random_data6 = data[:,index6]
random_image2 = random_data2.reshape((28,28)).transpose()
random_image6 = random_data6.reshape((28,28)).transpose()
plt.imshow(random_image2)
plt.show()
plt.imshow(random_image6)
plt.show()


# 2b
# calculate log-likelihood
def cal_ll(prob, pi):
    res = np.sum(np.log(prob@pi)) - N*P/2*np.log(2*math.pi)
    return res

iterN = 1
# here I choose to stop at certain iteration steps
# you can also choose to stop when it converges
while iterN < 50:
    print("Iteration:" + str(iterN))
    # E-step
    prob = gaussian(x, pi, mean, cov1, cov2)
    tau1 = pi[0]*prob[:,0]
    tau2 = pi[1]*prob[:,1]
    this_sum = tau1 + tau2
    tau1 = tau1/this_sum
    tau2 = tau2/this_sum
    tau = np.vstack((tau1, tau2)).T
    this_ll = cal_ll(prob, pi)
    ll.append(this_ll)
    
    # E-step
    pi = np.sum(tau, axis = 0)/N
    mean = data@tau/np.sum(tau, axis = 0)
    mean1 = np.reshape(mean[:,0], (P,1))
    mean2 = np.reshape(mean[:,1], (P,1))
    cov1 = tau[:,0] * (data-mean1) @ (data-mean1).T/np.sum(tau, axis = 0)[0]
    cov2 = tau[:,1] * (data-mean2) @ (data-mean2).T/np.sum(tau, axis = 0)[1]
    iterN += 1

plt.plot(np.arange(1,iterN),ll)
plt.show()
    
# 2c
if pi[0] > pi[1]:
    label2 = 0
    label6 = 1
else:
    label2 = 1
    label6 = 0
mean_2 = mean[:,label2].reshape((28,28)).transpose()
plt.imshow(mean_2)
plt.show()
mean_6 = mean[:,label6].reshape((28,28)).transpose()
plt.imshow(mean_6)
plt.show()

# 2d here you can just use packages to get k-means results
label_pred = np.zeros(N)
for i in range(N):
    if tau[i, label2] > tau[i, label6]:
        label_pred[i] = 2
    else:
        label_pred[i] = 6
print("EM-GMM Accuracy: " + str(sum(label_pred == label)/N))

# k-means
kmeans = KMeans(n_clusters=2).fit(x)
label_pred_kmeans = kmeans.labels_
for i in range(N):
    if label_pred_kmeans[i] == 0:
        label_pred_kmeans[i] = 2
    else:
        label_pred_kmeans[i] = 6
k_acc = max(sum(label_pred_kmeans == label)/N, 1-sum(label_pred_kmeans == label)/N)
print("K-means Accuracy: " + str(k_acc))


