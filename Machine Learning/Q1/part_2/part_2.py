import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Loading & Splitting Dataset
feature = np.loadtxt('data.dat').T
label = np.loadtxt('label.dat').astype(int)
label[label == 6] = 1
label[label == 2] = 0
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=1)

# Fitting models / Training
clf_svm = svm.SVC(kernel='linear', C = 0.01)
clf_svm.fit(X_train, y_train)

clf_nn = MLPClassifier(solver='sgd', learning_rate_init = 0.001, max_iter=1000, 
                       hidden_layer_sizes=(5, 2), random_state=1)
clf_nn.fit(X_train, y_train)

# Testing
predict_svm = clf_svm.predict(X_test)
acc_svm = np.average(predict_svm == y_test)
print('SVM Testing Accuracy:', acc_svm)

predict_nn = clf_nn.predict(X_test)
acc_nn = np.average(predict_nn == y_test)
print('NN Testing Accuracy:', acc_nn)

'''# Part (b)
# Taking first 2 features
X_train = X_train[:,:2]
X_test = X_test[:,:2]

# Fitting models / Training
clf_svm = svm.SVC(kernel='linear', C = 0.01)
clf_svm.fit(X_train, y_train)

clf_nn = MLPClassifier(solver='sgd', learning_rate_init = 0.1, max_iter=100, 
                       hidden_layer_sizes=(5, 2), random_state=1)
clf_nn.fit(X_train, y_train)

# Testing
predict_svm = clf_svm.predict(X_test)
acc_svm = np.average(predict_svm == y_test)
print('SVM Testing Accuracy:', acc_svm)

predict_nn = clf_nn.predict(X_test)
acc_nn = np.average(predict_nn == y_test)
print('NN Testing Accuracy:', acc_nn)

# Plotting

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

X_test_0 = X_test[y_test == 0]
X_test_1 = X_test[y_test == 1]



xx, yy = make_meshgrid(np.array([-2,4]), np.array([-1,4]), h=.02)

z_svm = clf_svm.predict(np.c_[xx.ravel(), yy.ravel()])
z_svm = z_svm.reshape(xx.shape)

z_nn = clf_nn.predict(np.c_[xx.ravel(), yy.ravel()])
z_nn = z_nn.reshape(xx.shape)

# SVM 
plt.figure()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.scatter(X_test_0[:,0], X_test_0[:,1], label = 'Class 0')
plt.scatter(X_test_1[:,0], X_test_1[:,1], label = 'Class 1')
plt.legend()
#plt.tricontour(X_test[:,0], X_test[:,1], np.array(predict_svm).astype(float), linewidths=0.5, colors='k')
#plt.contourf(xx, yy, z_svm,alpha = 0.5, cmap="RdBu_r")
plt.savefig('svm.png', dpi=600)

# NN
plt.figure()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.scatter(X_test_0[:,0], X_test_0[:,1], label = 'Class 0')
plt.scatter(X_test_1[:,0], X_test_1[:,1], label = 'Class 1')
plt.legend()
#plt.tricontour(X_test[:,0], X_test[:,1], np.array(predict_svm).astype(float), linewidths=0.5, colors='k')
#plt.contourf(xx, yy, z_nn,alpha = 0.5, cmap="RdBu_r")
plt.savefig('nn.png', dpi=600)

plt.show()'''
