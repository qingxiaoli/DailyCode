import numpy as np
import tensorflow as tf
import cifar10_input

# parameters
LEARNING_RATE = 1e-2
TRAINING_ITERS = 10000
BATCH_SIZE = 128
DISPLAY_STEP = 100

# network parameters
N_INPUT = cifar10_input.IMAGE_SIZE
N_CLASSES = cifar10_input.NUM_CLASSES
DROPOUT = 0.5

# create model
def conv_2d(img, w, b):
    return tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME') + b

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')

def evaluation(y_pred, y):
    correct = tf.equal(tf.argmax(y_pred, 1), tf.to_int64(y))
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
    with tf.variable_scope('conv_1'):
        conv1 = conv_2d(X, _weights['wc1'], _biases['bc1'])
        conv1 = batch_norm(conv1, train_phase=_train_phase)
        conv1 = tf.nn.relu(conv1)
        conv1 = max_pool(conv1, k=3)
        conv1 = tf.nn.dropout(conv1, keep_prob=_dropout)

    with tf.variable_scope('conv_2'):
        conv2 = conv_2d(conv1, _weights['wc2'], _biases['bc2'])
        conv2 = batch_norm(conv2, train_phase=_train_phase)
        conv2 = tf.nn.relu(conv2)
        conv2 = max_pool(conv2, k=3)
        conv2 = tf.nn.dropout(conv2, keep_prob=_dropout)

    with tf.variable_scope('fc1'):
        dense1 = tf.reshape(conv2, shape=[-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'])
        dense1 = tf.nn.dropout(dense1, keep_prob=_dropout)

    with tf.variable_scope('fc2'):
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'])
        dense2 = tf.nn.dropout(dense2, keep_prob=_dropout)

    with tf.variable_scope('output'):
        out = tf.nn.softmax(tf.matmul(dense2, _weights['out']) + _biases['out'])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(out), reduction_indices=[1]))
    loss = cross_entropy
    accuracy = evaluation(out, Y)
    return loss, accuracy, out

x = tf.placeholder(tf.float32, [None, N_INPUT, N_INPUT, 3])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
keep_prob = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.1), name='wc1'),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1), name='wc2'),
    'wd1': tf.Variable(tf.truncated_normal([6*6*64, 384], mean=0.0, stddev=0.05), name='wd1'),
    'wd2': tf.Variable(tf.truncated_normal([384, 192], mean=0.0, stddev=0.05), name='wd2'),
    'out': tf.Variable(tf.truncated_normal([192, N_CLASSES], mean=0.0, stddev=0.05), name='out')
}

biases = {
    'bc1': tf.Variable(tf.zeros([64]), name='bc1'),
    'bc2': tf.Variable(tf.zeros([64]), name='bc2'),
    'bd1': tf.Variable(tf.zeros([384]), name='bd1'),
    'bd2': tf.Variable(tf.zeros([192]), name='bd2'),
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

test_images, test_labels = cifar10_input.inputs(eval_data=1, data_dir='./cifar-10-batches-py', batch_size=BATCH_SIZE)
for i in range(TRAINING_ITERS):
    images_train, label_train = cifar10_input.distorted_inputs('./cifar-10-batches-py', BATCH_SIZE)
    optimizer.run(feed_dict={x: images_train.eval(), y: label_train.eval(), keep_prob: DROPOUT, train_phase: True})
    inner = feed_dict={x: images_train, y: label_train, keep_prob: DROPOUT, train_phase: True}
    if i % DISPLAY_STEP == 0:
        print('Iter:', str(i), 'Testing Accuracy:', sess.run(accuracy, feed_dict={x: test_images.eval(), y: test_labels.eval(), keep_prob: 1.0, train_phase: False}))