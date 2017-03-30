# this script is to do a classification with mnist data with cnn, by the way, try different optimizer
# coder: Jie An
# version: 20170327
# bug_submission: pkuanjie@gmail.com
# ====================================================================================================
import numpy as np
import tensorflow as tf
import mnist


# setup
MAX_TRAIN = 10000
BATCH_SIZE = 50
OPTIMIZER = 'adam'


# initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_22(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# import data
mnist_data = mnist.read_data_sets('MNIST_data/', one_hot=True)
train_x, train_y, test_x, test_y = mnist_data.train.images, mnist_data.train.labels, mnist_data.test.images,\
                                   mnist_data.test.labels

# create data holder
x = tf.placeholder('float', shape=[None, np.size(train_x, axis=1)])
y_ = tf.placeholder('float', shape=[None, np.size(train_y, axis=1)])
keep_prob = tf.placeholder('float')

# parameters creation
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# compute graph
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pooling_22(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pooling_22(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# train and test
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
if OPTIMIZER == 'sgd':
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
if OPTIMIZER == 'sgdm':
    train_step = tf.train.MomentumOptimizer(0.001, 0.0005).minimize(cross_entropy)
if OPTIMIZER == 'nest':
    train_step = tf.train.MomentumOptimizer(0.001, 0.0005, use_locking=True).minimize(cross_entropy)
if OPTIMIZER == 'adagrad':
    train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)
if OPTIMIZER == 'rms':
    train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)
if OPTIMIZER == 'adam':
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y_conv, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(MAX_TRAIN):
    batch = mnist_data.train.next_batch(BATCH_SIZE)
    if i % 100 == 0:
        test_accuracy = accuracy.eval(feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels, \
                                                 keep_prob: 1.0})
        print('step=', i, 'test_accuracy=', test_accuracy)
        if test_accuracy > 0.99:
            break
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
