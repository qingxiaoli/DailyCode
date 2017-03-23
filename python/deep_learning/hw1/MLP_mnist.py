# this script is to try a mnist digit number classification with MLP by using
# tensorflow
# coder: Jie An
# version: 20170322
# bug_submission: pkuanjie@gmail.com
# ==============================================================================
import numpy as np
import tensorflow as tf
import mnist


# set up HIDDEN_NEURON_NUM = 500
HIDDEN_NEURON_NUM = 500

# data input and preprocessing
mnist_data = mnist.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist_data.train.images, mnist_data.train.labels, mnist_data.test.images, mnist_data.test.labels

# weight init
W_h = tf.Variable(tf.random_normal([784, HIDDEN_NEURON_NUM], stddev=0.01))
W_o = tf.Variable(tf.random_normal([HIDDEN_NEURON_NUM, 10], stddev=0.01))

# construct model
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
Y_pre = tf.matmul(tf.nn.relu(tf.matmul(X, W_h)), W_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pre, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(Y_pre, 1)

# run graph
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        for start, end in zip(range(0, len(train_x), 128), range(128, len(train_x) + 1, 128)):
            sess.run(train_op, feed_dict={X: train_x[start: end], Y: train_y[start: end]})
        print('train num = ', i, 'accuracy = ', np.mean(np.argmax(test_y, axis=1) == sess.run(predict_op, feed_dict={X: test_x, Y: test_y})))

