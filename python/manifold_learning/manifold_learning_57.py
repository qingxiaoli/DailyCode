import numpy as np
import scipy.io as scio
from sklearn import decomposition as skdc
from matplotlib import pyplot
from sklearn import preprocessing


# preprocessing
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

# PCA
p = skdc.PCA(n_components=6)
p.fit(img_list)

print(p.components_)
mean_img = np.reshape(p.mean_, (w, h))
pyplot.imshow(mean_img)
pyplot.show()


# scaled PCA
img_list_scale = preprocessing.scale(img_list)
p_scale = skdc.PCA(n_components=6)
p_scale.fit(img_list)

print(p_scale.components_)
mean_img_scale = np.reshape(p_scale.mean_, (w, h))
pyplot.imshow(mean_img_scale)
pyplot.show()

