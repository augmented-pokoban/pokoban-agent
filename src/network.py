import tensorflow as tf
import numpy as np


def _conv2d(x, filters, scope, kernel_size, strides=2500, dropout=1.0):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [1, kernel_size, x.get_shape()[3], filters], tf.float32,
                            tf.random_normal_initializer(mean=0.001, stddev=0.02))
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')

        x = tf.nn.tanh(x) * tf.nn.sigmoid(x)
        print(x.get_shape())
        return x


def cnn(input_vector, dropout, input_shape):
    # How do we handle multi-dimensional input?
    x = tf.reshape(input_vector, shape=[-1, 1, input_shape[1], 1])

    conv = _conv2d(x, kernel_size=250, scope='conv1', strides=160, filters=250, dropout=dropout)
    conv = _conv2d(conv, kernel_size=250, scope='conv2', strides=160, filters=250, dropout=dropout)
    conv = _conv2d(conv, kernel_size=250, scope='conv3', strides=160, filters=250, dropout=dropout)

    return conv


def cnn_with_fc(input_vector, dropout, input_shape, output_num):
    conv = cnn(input_vector, dropout, input_shape)

    with tf.variable_scope('fully_connected'):
        #     # 14 * 14 * 32 is given by:
        #     # 14 = 28 / k where k is the k from the previous maxpooling layer, and 28 is the image dimensions
        #     # If there is multiple max pooling layers with k = 2, then we will get
        #     # 7 = 28 / k / k
        #     # 32 is given by the output of the previous layer

        # number of outputs in last layer:
        # no idea what this number should be, but it should be the sum of the units
        # This means:
        # * # of units in last layer: conv2d.shape[] ... somehow, x last_filters
        tensor_in_shape = conv.get_shape()
        conv = tf.reshape(conv, [-1, np.prod(tensor_in_shape[1:]).value])
        w = tf.get_variable('w', [np.prod(tensor_in_shape[1:]).value, output_num], tf.float32,
                            tf.random_normal_initializer(mean=0.001, stddev=0.02))

        conv = tf.matmul(conv, w)
        conv = tf.nn.relu(conv)

        return conv

