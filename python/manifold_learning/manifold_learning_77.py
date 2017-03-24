# this script is to compute and show result with different distance difinition of CMDS algorithm
# coder: Jie An
# version: 20170324
# bug_submission: pkuanjie@gmail.com
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as io

data = io.loadmat('umist_cropped.mat')['facedat'][0]
print(data[:, :, 0])
# plt.imshow(data[0])