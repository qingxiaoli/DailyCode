'''
    this script is to varify which distance is better for classify different type of face photos
    input:
        None
    output:
        None
    Coder: Jie An
    Version: 20170311
    Bug_submission: jie.an@pku.edu.cn
'''
import cv2
import numpy as np


def distance_compute(img1, img2, method):
    """this function is to compute distance between different images"""
    data1 = img1.reshape(112 * 92, 1)
    data2 = img2.reshape(112 * 92, 1)
    if method == 'Euclidean':
        return np.linalg.norm(data1 - data2)
    if method == 'lp':
        return np.sum(np.abs(data1 - data2))
    if method == 'Angular':
        return np.linalg.norm((data1 / np.linalg.norm(data1)) - (data2 / np.linalg.norm(data2)))
    if method == 'Cosine':
        return 1 - np.abs(np.dot(np.transpose(data1), data2) / (np.linalg.norm(data1) * np.linalg.norm(data2)))[0][0]


# image import and split
img = cv2.imread('umist_cropped.jpg', 0)
img = np.array(img)
img = img * 1.0 / 255
data = {}
for label in range(24):
    data[str(label)] = []
    for num in range(24):
        data[str(label)].append(img[num * 113: num * 113 + 112, label * 93: label * 93 + 92])

# inner and outer distance compute
count_inner = 0
count_outer = 0
distance_inner = np.array([0.0, 0.0, 0.0, 0.0])
distance_outer = np.array([0.0, 0.0, 0.0, 0.0])
for label1 in range(24):
    for num1 in range(24):
        for label2 in range(label1, 24):
            for num2 in range(24):
                if label1 == label2 and num1 < num2:
                    distance_inner[0] += distance_compute(data[str(label1)][num1], data[str(label2)][num2], 'Euclidean')
                    distance_inner[1] += distance_compute(data[str(label1)][num1], data[str(label2)][num2], 'lp')
                    distance_inner[2] += distance_compute(data[str(label1)][num1], data[str(label2)][num2], 'Angular')
                    distance_inner[3] += distance_compute(data[str(label1)][num1], data[str(label2)][num2], 'Cosine')
                    count_inner += 1
                if label1 != label2:
                    distance_outer[0] += distance_compute(data[str(label1)][num1], data[str(label2)][num2], 'Euclidean')
                    distance_outer[1] += distance_compute(data[str(label1)][num1], data[str(label2)][num2], 'lp')
                    distance_outer[2] += distance_compute(data[str(label1)][num1], data[str(label2)][num2], 'Angular')
                    distance_outer[3] += distance_compute(data[str(label1)][num1], data[str(label2)][num2], 'Cosine')
                    count_outer += 1

# result show
print('result show as follow:\n')
print('Euclidean diatance:', (distance_inner[0] / count_inner) / (distance_outer[0] / count_outer), '\n')
print('lp diatance:', (distance_inner[1] / count_inner) / (distance_outer[1] / count_outer), '\n')
print('Angular diatance:', (distance_inner[2] / count_inner) / (distance_outer[2] / count_outer), '\n')
print('Cosine diatance:', (distance_inner[3] / count_inner) / (distance_outer[3] / count_outer), '\n')
