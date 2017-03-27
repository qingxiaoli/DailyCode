import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio

# setup
DIM = 6
THRESHOLD = 12000

# import data
print('start import data from mat file')
data = sio.loadmat('umist_cropped.mat')['facedat'][0]
LENGTH = 0
for i in range(len(data)):
	LENGTH += np.size(data[i], axis=2)
img = np.zeros([np.size(data[0], axis=0) * np.size(data[0], axis=1), LENGTH])
count = 0
for i in range(len(data)):
    for j in range(np.size(data[i], axis=2)):
        img[:, count] = (np.reshape(data[i][:, :, j], [np.size(data[i], axis=0) * np.size(data[i], axis=1)]))
        count += 1
img = img * 1.0 / 255

# compute average face and centeralize
print('start computing average face and centeralize')
x = np.mean(img, axis=1)
img_c = np.zeros(np.shape(img))
for i in range(np.size(img, axis=1)):
	img_c[:, i] = img[:, i] - x

# generate the random matrix
print('start generate random matrix')
R = np.random.normal(0, 1, [DIM, np.size(img, axis=0)])
P = 1 / np.sqrt(DIM) * R
Y = np.dot(P, img_c)

# recognition faces
label = np.zeros(LENGTH, dtype=int) 
count = 0
for i in range(len(data)):
	for j in range(np.size(data[i], axis=2)):
		img_test = np.reshape(data[i][:, :, j], [np.size(img, axis=0)]) - x
		z = np.dot(P, img_test)
		tmp = np.zeros(np.size(Y, axis=1))
		for k in range(np.size(Y, axis=1)):
			tmp[k] = np.linalg.norm(Y[:, k] - z) ** 2
		value = np.sum(tmp)
		if value < THRESHOLD:
			label[count] = 1
		count += 1
print(label)
accuracy = 0
cut = 0
for i in range(len(data)):
	cut_out = cut + np.size(data[i], axis=2)
	for j in range(cut, cut_out):
		if label[j] >= cut and label[j] < cut_out:
			accuracy += 1
	cut = cut_out
accuracy = accuracy / len(label)
print(accuracy)