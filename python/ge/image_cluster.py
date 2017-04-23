import dicom
from matplotlib import pylab
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# data import and set up
data = dicom.read_file('./i1.CTDC.2')
img1 = data.pixel_array
img1 = img1.astype(float)
XBIN = range(1010, 1060, 5)
RADIUS = 205


def image_background_cut(img, radius):
    """cut image background which can't be used"""
    img_size = np.shape(img)
    X, Y = np.meshgrid(np.arange(0, img_size[0]), np.arange(0, img_size[1]))
    img[(X - img_size[0] / 2) ** 2 + (Y - img_size[1] / 2) ** 2 > radius ** 2] = 0
    return img


def create_circular_shape(img, radius_in, radius_out, max_random_intensity):
    """add a circular shape of random intensity on image"""
    img_size = np.shape(img)
    X, Y = np.meshgrid(np.arange(0, img_size[0]), np.arange(0, img_size[1]))
    img[((X - img_size[0] / 2) ** 2 + (Y - img_size[1] / 2) ** 2 > radius_in ** 2) & ((X - img_size[0] / 2) ** 2 + (Y - img_size[1] / 2) ** 2 < radius_out ** 2)] += \
        max_random_intensity * np.random.random(size=np.size(img[((X - img_size[0] / 2) ** 2 + (Y - img_size[1] / 2) ** 2 > radius_in ** 2) &\
                                                                 ((X - img_size[0] / 2) ** 2 + (Y - img_size[1] / 2) ** 2 < radius_out ** 2)]))
    return img


def circular_hist_analysis(img_piece, xbin=XBIN):
    """compute hist array of image piece """
    counts, centers = pylab.histogram(img_piece, xbin, normed=True)
    counts = np.array(counts)
    counts.shape = (len(centers) - 1, 1)
    counts = np.transpose(counts)
    return counts

# circular image piece histogram compute
img1 = image_background_cut(img1, RADIUS)
img1 = create_circular_shape(img1, 30, 50, 20)
PIECE_RADIUS = 20
hist_data = np.zeros([len(range(0, RADIUS, PIECE_RADIUS)), len(XBIN) - 1])
for i in range(0, RADIUS, PIECE_RADIUS):
    img_size = np.shape(img1)
    X, Y = np.meshgrid(np.arange(0, img_size[0]), np.arange(0, img_size[1]))
    img_piece = img1[((X - img_size[0] / 2) ** 2 + (Y - img_size[1] / 2) ** 2 > i ** 2) & ((X - img_size[0] / 2) ** 2 + (Y - img_size[1] / 2) ** 2 < (i + PIECE_RADIUS) ** 2)]
    hist_data[int(i / PIECE_RADIUS), :] = circular_hist_analysis(img_piece)

# data classification with sklearn.svm
hist_label = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
scaler = StandardScaler().fit(hist_data) #data preprocessing, normalization
hist_data = scaler.transform(hist_data)
clf = svm.SVC()
clf.fit(hist_data, hist_label)
for i in range(len(hist_label)):
    predicted_label = clf.predict(hist_data[i, :].reshape(1, -1))
    print('predicted label of No.', i, 'piece image is', predicted_label)
