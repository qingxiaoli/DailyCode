import numpy as np
import sklearn.manifold as skm
import scipy.io as io
from matplotlib import pyplot as plt


# import data from mat file and setup
data = io.loadmat('umist_cropped.mat')['facedat'][0]
data = data * 1.0 / 255
img = []
for i in range(len(data)):
    for j in range(np.size(data[i], axis=2)):
        img.append(np.reshape(data[i][:, :, j], [1, np.size(data[i], axis=0) * np.size(data[i], axis=1)])[0])
img = np.array(img)

# MDS
Y = skm.MDS().fit_transform(img)
Y = np.transpose(Y)
f1 = plt.figure(1)
plt.title('MDS with lp norm')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
count = 0
for i in range(len(data)):
    count_out = count + np.size(data[i], axis=2)
    plt.scatter(Y[1, count: count_out], Y[0, count: count_out], c=colors[i])
    count = count_out

# isomap
Y = skm.Isomap().fit_transform(img)
Y = np.transpose(Y)
f2 = plt.figure(2)
plt.title('ISOMAP with lp norm')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
count = 0
for i in range(len(data)):
    count_out = count + np.size(data[i], axis=2)
    plt.scatter(Y[1, count: count_out], Y[0, count: count_out], c=colors[i])
    count = count_out

# LLE 
Y = skm.LocallyLinearEmbedding().fit_transform(img)
Y = np.transpose(Y)
f3 = plt.figure(3)
plt.title('LLE with lp norm')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
count = 0
for i in range(len(data)):
    count_out = count + np.size(data[i], axis=2)
    plt.scatter(Y[1, count: count_out], Y[0, count: count_out], c=colors[i])
    count = count_out

# LTSA 
Y = skm.LocallyLinearEmbedding(method='ltsa', eigen_solver='dense').fit_transform(img)
Y = np.transpose(Y)
f4 = plt.figure(4)
plt.title('LTSA with lp norm')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
count = 0
for i in range(len(data)):
    count_out = count + np.size(data[i], axis=2)
    plt.scatter(Y[1, count: count_out], Y[0, count: count_out], c=colors[i])
    count = count_out

# LE 
Y = skm.SpectralEmbedding().fit_transform(img)
Y = np.transpose(Y)	
f5 = plt.figure(5)
plt.title('LE with lp norm')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
count = 0
for i in range(len(data)):
    count_out = count + np.size(data[i], axis=2)
    plt.scatter(Y[1, count: count_out], Y[0, count: count_out], c=colors[i])
    count = count_out
plt.show()