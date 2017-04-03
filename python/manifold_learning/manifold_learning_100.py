import numpy as np
import scipy.io as scio


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
img_list -= np.mean(img_list, axis=0)
img_list = np.transpose(img_list)
X = np.mat(img_list)
one_1 = np.ones([np.size(img_list, axis=1), 1])
one_1 = np.mat(one_1)

# w_0 compute
l = (np.dot(np.dot(one_1.T, np.dot(X.T, X).I), np.dot(X.T, X[:, 0])) - 1) / (np.dot(np.dot(one_1.T, np.dot(X.T, X).I), \
                                                                                    one_1))
l = np.array(l)[0][0]
a = np.dot(np.dot(X.T, X).I, (np.dot(X.T, X[:, 0]) - l * one_1))

print(a)