from autoencoder.EncoderData import DataLoader, batch_to_lists
from autoencoder.EncoderNetwork import EncoderNetwork
from env import api
from env.Env import Env, matrix_to_state, State
import tensorflow as tf
import numpy as np

from helper import reshape_back

batch_size = 1
height = 20
width = 20
depth = 1
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
f_size = 7

data = DataLoader('../data.csv', batch_size, '../batches/')

model_path = './enc_model'

tf.reset_default_graph()
trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, f_size, batch_size, 'global', trainer)
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    # ckpt = tf.train.get_checkpoint_state(model_path)
    # saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(tf.global_variables_initializer())

    x_state, x_action, exp_state, exp_reward = batch_to_lists(data.get_train(), s_size)
    y_reward, y_state = network.eval(x_state, x_action, sess)

    print(y_state)
    exit(0)
    diff = np.absolute(exp_state - y_state)
    missing = exp_state - y_state
    missing = np.clip(missing, 0, 1)

    overfit = y_state - exp_state
    overfit = np.clip(overfit, 0, 1)

    result = dict()
    exp_reward = exp_reward[0, 0]
    y_reward = np.argmax(y_reward[0])

    print(np.sum(diff))

    result['errors'] = np.sum(diff)
    result['missing_errors'] = np.sum(missing)
    result['overfit_errors'] = np.sum(overfit)
    result['x_state'] = matrix_to_state(reshape_back(x_state, height, width, depth), 20)
    result['y_state_exp'] = matrix_to_state(reshape_back(exp_state, height, width, depth), 20)
    result['y_state_eval'] = matrix_to_state(reshape_back(y_state, height, width, depth), 20)
    result['diff'] = matrix_to_state(reshape_back(diff, height, width, depth), 20)
    result['y_reward_exp'] = Env.get_reward_meanings()[exp_reward]
    result['y_reward_eval'] = Env.get_reward_meanings()[y_reward]
    result['action'] = Env.get_action_meanings()[x_action[0, 0]]
    result['success'] = bool(exp_reward == y_reward)
    result['missing'] = matrix_to_state(reshape_back(missing, height, width, depth), 20)
    result['overfit'] = matrix_to_state(reshape_back(overfit, height, width, depth), 20)

    api.post_encoding(result)
