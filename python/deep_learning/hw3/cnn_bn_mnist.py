import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

# parameters
LEARNING_RATE = 1e-2
TRAINING_ITERS = 10000
BATCH_SIZE = 128
DISPLAY_STEP = 100

# network parameters
N_INPUT = 784
N_CLASSES = 10
DROPOUT = 0.5

# create model
def conv_2d(img, w, b):
	return tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME') + b

def max_pool(img, k):
	return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def evaluation(y_pred, y):
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return accuracy

def batch_norm(X, train_phase):
	with tf.name_scope('bn'):
		n_out = X.get_shape()[-1:]
		beta = tf.Variable(tf.constant(0.0, shape=n_out), name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=n_out), name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(train_phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
	return normed

def conv_net(X, Y, _weights, _biases, _dropout, _train_phase):
	_X = tf.reshape(X, shape=[-1, 28, 28, 1])
	with tf.variable_scope('conv_1'):
		conv1 = conv_2d(_X, _weights['wc1'], _biases['bc1'])
		conv1 = batch_norm(conv1, train_phase=_train_phase)
		conv1 = tf.nn.relu(conv1)
		conv1 = max_pool(conv1, k=2)
		conv1 = tf.nn.dropout(conv1, keep_prob=_dropout)

	with tf.variable_scope('conv_2'):
		conv2 = conv_2d(conv1, _weights['wc2'], _biases['bc2'])
		conv2 = batch_norm(conv2, train_phase=_train_phase)
		conv2 = tf.nn.relu(conv2)
		conv2 = max_pool(conv2, k=2)
		conv2 = tf.nn.dropout(conv2, keep_prob=_dropout)

	with tf.variable_scope('fc'):
		dense1 = tf.reshape(conv2, shape=[-1, _weights['wd1'].get_shape().as_list()[0]])
		dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'])
		dense1 = tf.nn.dropout(dense1, keep_prob=_dropout)

	with tf.variable_scope('output'):
		out = tf.nn.softmax(tf.matmul(dense1, _weights['out']) + _biases['out'])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(out), reduction_indices=[1]))
	loss = cross_entropy
	accuracy = evaluation(out, Y)
	return loss, accuracy, out

x = tf.placeholder(tf.float32, [None, N_INPUT])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
keep_prob = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

weights = {
	'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='wc1'),
	'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='wc2'),
	'wd1': tf.Variable(tf.truncated_normal([7*7*64, 1024], mean=0.0, stddev=0.05), name='wd1'),
	'out': tf.Variable(tf.truncated_normal([1024, N_CLASSES], mean=0.0, stddev=0.05), name='out')
}

biases = {
	'bc1': tf.Variable(tf.zeros([32]), name='bc1'),
	'bc2': tf.Variable(tf.zeros([64]), name='bc2'),
	'bd1': tf.Variable(tf.zeros([1024]), name='bd1'),
	'out': tf.Variable(tf.zeros([N_CLASSES]), name='out')
}

wc1_hist = tf.summary.histogram('wc1', weights['wc1'])
wc2_hist = tf.summary.histogram('wc2', weights['wc2'])
wd1_hist = tf.summary.histogram('wd1', weights['wd1'])
wout_hist = tf.summary.histogram('weights_out', weights['out'])

bc1_hist = tf.summary.histogram('bc1', biases['bc1'])
bc2_hist = tf.summary.histogram('bc2', biases['bc2'])
bd1_hist = tf.summary.histogram('bd1', biases['bd1'])
bout_hist = tf.summary.histogram('biases_out', biases['out'])

y_hist = tf.summary.histogram('y', y)

loss, accuracy, pred = conv_net(x, y, weights, biases, keep_prob, train_phase)
optimizer = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(loss)
cost_summary = tf.summary.scalar('loss', loss)
acc_summary = tf.summary.scalar('accuracy', accuracy)
merge_summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(TRAINING_ITERS):
	batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
	optimizer.run(feed_dict={x: batch_xs, y: batch_ys, keep_prob: DROPOUT, train_phase: True})
	inner = feed_dict={x: batch_xs, y: batch_ys, keep_prob: DROPOUT, train_phase: True}
	if i % DISPLAY_STEP == 0:
		# acc = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0, train_phase: False})
		# error = loss.eval(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0, train_phase: False})
		# print('Iter ' + str(i) + ' Minibatch Loss= ' + '{:.6f}'.format(error) + ', Training Accuracy= ' + '{:.5f}'.format(acc))
		print('Iter:', str(i), 'Testing Accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, train_phase: False}))
