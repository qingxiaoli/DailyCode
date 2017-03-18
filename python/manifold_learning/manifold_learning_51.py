import scipy.io as scio
import numpy as np
from sklearn.cluster import SpectralClustering as SC


# prepossing
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

W = np.zeros((len(img_list), len(img_list)))
for i in range(len(img_list)):
    for j in range(i + 1, len(img_list)):
       W[i, j] = np.linalg.norm(img_list[i] - img_list[j])
       W[j, i] = W[i, j]

mean1 = 0
mean2 = 0
for i in range(shape_img0):
    for j in range(i + 1, shape_img0):
        mean1 += W[i, j]
for i in range(shape_img0, len(img_list)):
    for j in range(i + 1, len(img_list)):
        mean2 += W[i, j]

mean1 /= (shape_img0 * (shape_img0 - 1) / 2)
mean2 /= (shape_img1 * (shape_img1 - 1) / 2)

sigma1 = 0
sigma2 = 0
for i in range(shape_img0):
    for j in range(i + 1, shape_img0):
        sigma1 += (W[i, j] - mean1) ** 2
for i in range(shape_img0, len(img_list)):
    for j in range(i + 1, len(img_list)):
        sigma2 += (W[i, j] - mean2) ** 2

sigma1 = np.sqrt(sigma1 / (shape_img0 * (shape_img0 - 1) / 2))
sigma2 = np.sqrt(sigma2 / (shape_img1 * (shape_img1 - 1) / 2))
sigma = (sigma1 + sigma2) / 2

W = np.exp(-1 * W / sigma)

# NCut cluster
img_list = np.array(img_list)
cluster = SC(n_clusters=2, affinity='precomputed')
cluster.fit(W)
result = cluster.fit_predict(W)
print(result)

# accuracy compute
accuracy = 0
for i in range(len(result)):
    if i < shape_img0 and result[i] == 0:
        accuracy += 1
    if i >= shape_img0 and result[i] == 1:
        accuracy += 1
accuracy = float(accuracy) / len(result)
print(accuracy)
