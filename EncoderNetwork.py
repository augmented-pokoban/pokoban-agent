import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from helper import normalized_columns_initializer


class EncoderNetwork:

    def __init__(self, height, width, depth, s_size, scope):

        with tf.variable_scope(scope):

            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

            self.imageIn = tf.reshape(self.inputs, shape=[-1, height, width, depth])

            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=16,
                                     kernel_size=[3, 3], stride=[1, 1], padding='VALID')

            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[3, 3], stride=[2, 2], padding='VALID')

            action = 0

            enc_out = slim.fully_connected(
                tf.concat(slim.flatten(self.conv2), [action]),
                256,
                activation_fn=tf.nn.elu
            )

            # Output layers for encoding and value estimations
            self.encoding = slim.fully_connected(enc_out, s_size,
                                                 activation_fn=tf.nn.sigmoid,
                                                 weights_initializer=normalized_columns_initializer(0.5),
                                                 biases_initializer=None)

            # value = reward
            self.value = slim.fully_connected(enc_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            self.enc_target = None
            self.val_target = None

            # Loss functions
            self.encoding_loss = tf.reduce_mean(tf.squared_difference(self.encoding, self.enc_target))
            self.value_loss = -tf.reduce_sum(tf.square(self.value, self.val_target))

            self.loss = self.value_loss + self.encoding_loss

            l_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, l_vars)
            self.var_norms = tf.global_norm(l_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
