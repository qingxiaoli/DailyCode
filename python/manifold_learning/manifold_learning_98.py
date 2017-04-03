import numpy as np
from sklearn.manifold import Isomap
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

# isomap
Y = Isomap().fit_transform(img)
Y = np.transpose(Y)


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
