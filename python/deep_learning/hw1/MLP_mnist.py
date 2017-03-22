import numpy as np
import tensorflow as tf
import input_data


# set up
HIDDEN_NEURON_NUM = 500

# data input and preprocessing
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist_data.train.images, mnist_data.train \
    .labels, mnist_data.test.images, mnist_data.test.labels

# weight init
W_h = tf.Variable(tf.random_normal([768, HIDDEN_NEURON_NUM], stddev=0.01))
W_o = tf.Variable(tf.random.normal([HIDDEN_NEURON_NUM, 10], stddev=0.01))

# construct model
X = tf.placeholder('float', [None, 768])
Y = tf.placeholder('float', [None, 10])
model_mlp = tf.matmul(tf.nn.relu(tf.matmul(X, W_h)), W_o)
