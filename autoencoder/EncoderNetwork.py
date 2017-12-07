import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from helper import normalized_columns_initializer


class EncoderNetwork:

    def __init__(self, height, width, depth, s_size, a_size, r_size, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.input_image = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.action = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.enc_target = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.val_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            # one-hot action vector

            self.actions_onehot = tf.one_hot(self.action, a_size, dtype=tf.float32)
            self.ones_placeholder = tf.placeholder(shape=[None, height, width, a_size], dtype=tf.float32)

            # combine input one-hot with action tile one-hot
            actions_tile = tf.ones_like(self.ones_placeholder) * self.actions_onehot
            self.input_tile = tf.concat([self.input_image, actions_tile], axis=-1)

            self.imageIn = tf.reshape(self.input_tile, shape=[-1, height, width, depth + a_size])

            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=16,
                                     kernel_size=[3, 3], stride=[1, 1], padding='VALID')

            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[3, 3], stride=[2, 2], padding='VALID')

            enc_out = slim.fully_connected(
                slim.flatten(self.conv2),
                256,
                activation_fn=tf.nn.elu
            )

            # Output layers for encoding and value estimations
            self.encoding = slim.fully_connected(enc_out, s_size,
                                                 activation_fn=tf.nn.sigmoid,
                                                 weights_initializer=normalized_columns_initializer(0.5),
                                                 biases_initializer=None)

            # value = reward
            self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv2, num_outputs=32,
                                     kernel_size=[1, 1], stride=[1, 1], padding='VALID')
            self.conv4 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv3, num_outputs=32,
                                     kernel_size=[1, 1], stride=[1, 1], padding='VALID')

            self.value = slim.fully_connected(self.conv4,
                                              r_size,
                                              activation_fn=tf.nn.softmax,
                                              weights_initializer=normalized_columns_initializer(0.01),
                                              biases_initializer=None)

            # Loss functions
            self.encoding_loss = tf.reduce_mean(tf.squared_difference(self.encoding, self.enc_target))
            self.value_loss = -tf.reduce_sum(tf.square(self.value, self.val_target))

            self.loss = self.encoding_loss + self.value_loss

            self.train_op = trainer.minimize(self.loss)
