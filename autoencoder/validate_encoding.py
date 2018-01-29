from autoencoder.EncoderData import DataLoader, batch_to_lists
from autoencoder.EncoderNetwork import EncoderNetwork
from env import api
from env.Env import Env, matrix_to_state, State, new_matrix_to_state
import tensorflow as tf
import numpy as np

from helper import reshape_back

batch_size = 5
height = 20
width = 20
depth = 1
s_size = height * width * depth
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
f_size = 7

data = DataLoader('../data.csv', batch_size, '../batches/')

model_path = '../encoder_model_v1'

tf.reset_default_graph()
trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, f_size, batch_size, 'global', trainer)
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    # sess.run(tf.global_variables_initializer())

    # [s_size] in new matrix form for both x_state and exp_state
    x_state, x_action, exp_state, exp_reward = batch_to_lists(data.get_train(), s_size)
    y_reward, y_state = network.eval(x_state, x_action, sess)

    for i in range(batch_size):
        # y_state is [s_size] in rounded, new matrix form

        diff = np.absolute(exp_state[i] - y_state[i])
        diff = np.where(diff == 0, diff, 1.0)

        # print(x_state.shape)
        # print(exp_state.shape)
        # print(y_state.shape)
        # print(diff.shape)

        # missing = exp_state - y_state
        # missing = np.clip(missing, 0, 1)
        #
        # overfit = y_state - exp_state
        # overfit = np.clip(overfit, 0, 1)

        result = dict()
        exp_reward = exp_reward[i]
        print(y_reward)
        exit(0)
        y_reward = np.argmax(y_reward[0][i])

        print(np.sum(diff))

        result['errors'] = np.sum(diff)
        # result['missing_errors'] = np.sum(missing)
        # result['overfit_errors'] = np.sum(overfit)
        result['x_state'] = new_matrix_to_state(reshape_back(x_state[i], height, width), 20)
        result['y_state_exp'] = new_matrix_to_state(reshape_back(exp_state[i], height, width), 20)
        result['y_state_eval'] = new_matrix_to_state(reshape_back(y_state[0], height, width), 20)
        result['diff'] = new_matrix_to_state(reshape_back(diff[0], height, width), 20)
        result['y_reward_exp'] = Env.get_reward_meanings()[exp_reward]
        result['y_reward_eval'] = Env.get_reward_meanings()[y_reward]
        result['action'] = Env.get_action_meanings()[x_action[i, 0]]
        result['success'] = bool(exp_reward == y_reward)
        # result['missing'] = matrix_to_state(reshape_back(missing, height, width, depth), 20)
        # result['overfit'] = matrix_to_state(reshape_back(overfit, height, width, depth), 20)

        api.post_encoding(result)
