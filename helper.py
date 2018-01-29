import tensorflow as tf
import numpy as np


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Resize and grayscale the image
def process_frame(s, s_size):
    # r, g, b = s[:, :, 0], s[:, :, 1], s[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    # img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b
    # img_gray = scipy.misc.imresize(img_gray, [height, width])

    img_gray = np.reshape(s, [s_size])
    return img_gray


def reshape_back(s, height, width):
    return np.reshape(s, [height, width])


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    rewards = np.empty(len(x))
    cum_reward = 0

    for step in reversed(range(len(x))):
        cum_reward = x[step] + cum_reward * gamma
        rewards[step] = cum_reward
    return rewards

#
# def discount_old(x, gamma):
#     return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer
