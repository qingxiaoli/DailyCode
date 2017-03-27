# this script is to compute and show result with different distance definition of CMDS algorithm
# coder: Jie An
# version: 20170324
# bug_submission: pkuanjie@gmail.com
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as io

# setup
DIM = 2

# distance compute function
def distance_compute(img, i, j, method='Eculidean'):
    """ this function is to compute distance between two data with different methods"""
    if method == 'Euclidean':
        return np.linalg.norm(img[i] - img[j])
    if method == 'lp':
        return np.sum(np.abs(img[i] - img[j]))
    if method == 'Angular':
        return np.linalg.norm((img[i] / np.linalg.norm(img[i])) - (img[j] / np.linalg.norm(img[j])))
    if method == 'Cosine':
        return 1 - np.abs(np.dot(np.transpose(img[i]), img[j]) / (np.linalg.norm(img[i]) * np.linalg.norm(img[j])))[0][0]

# import data from mat file and setup
data = io.loadmat('umist_cropped.mat')['facedat'][0]
data = data * 1.0 / 255
img = []
for i in range(len(data)):
    for j in range(np.size(data[i], axis=2)):
        img.append(np.reshape(data[i][:, :, j], [1, np.size(data[i], axis=0) * np.size(data[i], axis=1)]))

# compute distance matrix
print('start computing distance matrix')
D = np.zeros([len(img), len(img)])
for i in range(len(img)):
    for j in range(i + 1, len(img)):
       D[i, j] = distance_compute(img, i, j, method='lp')
       D[j, i] = D[i, j]

# compute Y
print('start computing projection matrix')
S = D * D
S_c = np.zeros([np.size(S, axis=0), np.size(S, axis=1)])
for i in range(np.size(D, axis=0)):
    for j in range(np.size(D, axis=1)):
        S_c[i, j] = S[i, j] - np.mean(S[:, j]) - np.mean(S[i, :]) + np.mean(S)
G_c = -1 / 2 * S_c
w, v = np.linalg.eigh(G_c)
U = v[:, np.size(v, axis=1) - DIM: np.size(v, axis=1)]
Lambda = np.diag(w[len(w) - DIM: len(w)])
Y = np.dot(np.sqrt(Lambda), np.transpose(U))

# plot points in 2D
f1 = plt.figure(1)
plt.title('lp')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
count = 0
for i in range(len(data)):
    count_out = count + np.size(data[i], axis=2)
    plt.scatter(Y[1, count: count_out], Y[0, count: count_out], c=colors[i])
    count = count_out
plt.show()
