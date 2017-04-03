import numpy as np
import scipy.io as scio
from sklearn import decomposition as skdc


# set up
EPSILON1 = 50
EPSILON2 = 30000

# pre processing
data = scio.loadmat('umist_cropped.mat')
img = data['facedat'][0][0: 2]
img = img * 1.0 / 255
w = img[0].shape[0]
h = img[0].shape[1]
shape_img0 = img[0].shape[2]
shape_img1 = img[1].shape[2]
img_list = []

for i in range(shape_img0):
    img_list.append(np.reshape(img[0][:, :, i], (1, w * h))[0])

for i in range(shape_img1):
    img_list.append(np.reshape(img[1][:, :, i], (1, w * h))[0])

img_list = np.array(img_list)
img_list = np.transpose(img_list)
img_train = img_list[:, 0: int(np.size(img_list, axis=1) * 3 / 4)]
img_test = img_list[:, int(np.size(img_list, axis=1) * 3 / 4): np.size(img_list, axis=1)]

# PCA
for i in range(1, 6):
    p = skdc.KernelPCA(n_components=6, kernel="linear", gamma=1, degree=i)
    x = np.mean(img_train, axis=1)
    for k in range(np.size(img_train, axis=1)):
        img_train[:, k] -= np.transpose(x)
    p.fit(img_train)
    F = p.alphas_
    Y = np.dot(np.transpose(F), img_train)
    tmp = np.zeros(np.shape(img_test))
    for k in range(np.size(img_test, axis=1)):
        tmp[:, k] = img_test[:, k] - np.transpose(x)
    z = np.dot(np.transpose(F), tmp)
    tmp2 = np.zeros(np.shape(np.dot(F, z)))
    for k in range(np.size(tmp2, axis=1)):
        tmp2[:, k] = np.dot(F, z)[:, k] + np.transpose(x)
    d_1 = np.linalg.norm(img_test - tmp2, axis=0)
    d_2 = []
    for k in range(np.size(img_test, axis=1)):
        d = 0
        for j in range(np.size(img_train, axis=1)):
            d += np.linalg.norm(z[:, k] - Y[:, j]) ** 2
        d_2.append(d)
    accuracy = 0
    for k in range(len(d_2)):
        if d_1[k] < EPSILON1 and d_2[k] < EPSILON2:
            accuracy += 1
    accuracy /= len(d_2)
    print('degree of polynomial=', i, 'accuracy=', accuracy)
