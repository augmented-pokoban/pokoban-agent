from autoencoder.EncoderData import DataLoader, batch_to_lists, load_object
from autoencoder.EncoderNetwork import EncoderNetwork
from env.Env import Env
from env.mapper import new_matrix_to_state
import tensorflow as tf
import numpy as np

from helper import reshape_back
from support.post_state_diff import save_state_diff

batch_size = 1
height = 20
width = 20
depth = 1
s_size = height * width * depth
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
f_size = 7

# data = DataLoader('../data_test.csv', batch_size, '../batches/')
data = np.asarray(load_object('../validate_rollout/5_rollouts.pkl.zip'))
model_path = '../encoder_model_v2'

tf.reset_default_graph()
trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, f_size, batch_size, 'global', trainer)
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    # sess.run(tf.global_variables_initializer())

    # val_set = data.get_val()
    val_set = data[97, :]
    y_state = None

    for rollout in range(5):
        # [s_size] in new matrix form for both x_state and exp_state
        # x_state, x_action, exp_state, exp_reward, exp_success = batch_to_lists(val_set, s_size)
        x_state, x_action, exp_state, exp_reward, exp_success = batch_to_lists([val_set[rollout]], s_size)

        if rollout != 0:
            x_state = y_state

        y_reward, y_state, y_success = network.eval(x_state, x_action, sess)

        for i in range(batch_size):
            # y_state is [s_size] in rounded, new matrix form

            diff = np.absolute(exp_state[i] - y_state[i])
            diff = np.where(diff == 0, diff, 1.0)

            # print(x_state.shape)
            # print(exp_state.shape)
            # print(y_state.shape)
            # print(diff.shape)

            exp_r = exp_reward[i, 0]

            save_state_diff(exp_r=Env.get_reward_meanings()[exp_r],
                            act_r=Env.get_reward_meanings()[np.argmax(y_reward[i])],
                            diff=new_matrix_to_state(reshape_back(diff, height, width), 20),
                            errors=np.sum(diff),
                            x_state=new_matrix_to_state(reshape_back(x_state[i], height, width), 20),
                            y_state_exp=new_matrix_to_state(reshape_back(exp_state[i], height, width), 20),
                            y_state_act=new_matrix_to_state(reshape_back(y_state[i], height, width), 20),
                            action=x_action[i, 0],
                            success=bool(exp_success[i, 0] == np.argmax(y_success[i]))
                            )

            print('Submitted {}'.format(i + 1))
