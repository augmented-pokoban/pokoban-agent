import tensorflow as tf
import tensorflow.contrib.slim as slim

from helper import normalized_columns_initializer


class EncoderNetwork:

    def __init__(self, height, width, depth, s_size, a_size, r_size, batch_size, scope, trainer):
        with tf.variable_scope(scope):
            # episode counter
            self.episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

            # Input and visual encoding layers
            self.input_image = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.action = tf.placeholder(shape=[None, 1], dtype=tf.int32)
            self.enc_target = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.reward = tf.placeholder(shape=[None, 1], dtype=tf.int32)

            # one-hot action vector

            self.val_target = tf.one_hot(self.reward, r_size, dtype=tf.float32)

            self.actions_onehot = tf.one_hot(self.action, a_size, dtype=tf.int32)
            # self.actions_onehot = tf.cast(self.actions_onehot, tf.float32)
            self.actions_onehot = tf.expand_dims(self.actions_onehot, axis=2)
            self.ones_placeholder = tf.placeholder(shape=[batch_size, height, width, a_size], dtype=tf.int32)

            # combine input one-hot with action tile one-hot
            actions_tile = tf.ones_like(self.ones_placeholder) * self.actions_onehot
            actions_tile = tf.cast(actions_tile, tf.float32)

            self.imageIn = tf.reshape(self.input_image, shape=[-1, height, width, depth])
            self.input_tile = tf.concat([self.imageIn, actions_tile], axis=-1)

            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.input_tile, num_outputs=64,
                                     kernel_size=[3, 3], stride=[1, 1], padding='VALID')

            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[3, 3], stride=[1, 1], padding='VALID')

            enc_out = slim.fully_connected(
                slim.flatten(self.conv2),
                256,
                activation_fn=tf.nn.elu
            )

            # Output layers for encoding and value estimations
            self.encoding = slim.fully_connected(enc_out, s_size,
                                                 activation_fn=tf.nn.sigmoid, # potentionally chance this activation function
                                                 weights_initializer=normalized_columns_initializer(0.5),
                                                 biases_initializer=None)

            # value = reward
            self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv2, num_outputs=32,
                                     kernel_size=[1, 1], stride=[1, 1], padding='VALID')

            self.conv4 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv3, num_outputs=32,
                                     kernel_size=[1, 1], stride=[1, 1], padding='VALID')

            val_out = slim.fully_connected(
                slim.flatten(self.conv4),
                512,
                activation_fn=tf.nn.elu
            )

            self.value = slim.fully_connected(val_out,
                                              1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            self.encoding_rounded = tf.round(self.encoding)

            # Loss functions - mean squared error
            self.encoding_loss = tf.reduce_mean(tf.squared_difference(self.encoding, self.enc_target))
            self.value_loss = tf.reduce_mean(tf.square(self.value - self.val_target))

            self.loss = self.encoding_loss + 0.8 * self.value_loss

            self.rounded_loss = tf.reduce_mean(tf.squared_difference(self.encoding_rounded, self.enc_target))

            self.train_op = trainer.minimize(self.loss)

    def train(self, x_state, x_action, y_state, y_reward, sess):
        feed_dict = {
            self.input_image: x_state,
            self.action: x_action,
            self.enc_target: y_state,
            self.reward: y_reward
        }

        _, enc_loss, val_loss = sess.run(
            [
                self.train_op,
                self.encoding_loss,
                self.value_loss
            ],
            feed_dict=feed_dict
        )

        return enc_loss, val_loss

    def test(self, x_state, x_action, y_state, y_reward, sess):
        feed_dict = {
            self.input_image: x_state,
            self.action: x_action,
            self.enc_target: y_state,
            self.reward: y_reward
        }

        test_enc_loss, test_val_loss, test_rounded_loss = sess.run(
            [
                self.encoding_loss,
                self.value_loss,
                self.rounded_loss
            ],
            feed_dict=feed_dict
        )

        return test_enc_loss, test_val_loss, test_rounded_loss

    def eval(self, x_state, x_action, sess):

        feed_dict = {
            self.input_image: x_state,
            self.action: x_action
        }

        val, y = sess.run([self.value, self.encoding_rounded], feed_dict=feed_dict)

        return val, y



