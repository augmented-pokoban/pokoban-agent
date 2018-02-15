from autoencoder.EncoderData import DataLoader, batch_to_lists, load_object, batch_to_lists_preprocessed
from autoencoder.EncoderNetwork import EncoderNetwork
from env.Env import Env
from env.mapper import new_matrix_to_state
import tensorflow as tf
import numpy as np

from helper import reshape_back
from support.post_state_diff import save_state_diff

batch_size = 2000
height = 20
width = 20
depth = 1
s_size = height * width * depth
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
f_size = 7

model_path = '../encoder_model_v2'
data_path = '../succ_fail_plays/'
fail_path = '{}failures.pkl.zip'.format(data_path)
succ_path = '{}successes.pkl.zip'.format(data_path)

# read from zip files
data = load_object(fail_path) + load_object(succ_path)

print('Data len: {}'.format(len(data)))
# exp_act
succ_succ = 0
succ_fail = 0
fail_fail = 0
fail_succ = 0

tf.reset_default_graph()
trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, f_size, batch_size, 'global', trainer)
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # [s_size] in new matrix form for both x_state and exp_state
    x_state, x_action, exp_state, exp_reward, exp_success = batch_to_lists_preprocessed(data, s_size)
    y_reward, y_state, y_success = network.eval(x_state, x_action, sess)

    for i in range(batch_size):
        # y_state is [s_size] in rounded, new matrix form

        exp_r = exp_success[i, 0]
        act_r = np.argmax(y_success[i])

        if exp_r == 1 and act_r == 1:
            succ_succ += 1
        elif exp_r == 1 and act_r != 1:
            succ_fail += 1
        elif exp_r != 1 and act_r != 1:
            fail_fail += 1
        else:
            fail_succ += 1

print('\t\t\tExpected')
print('\t\t\tSucc.\tFail.')
print('pred. Succ:\t{}\t{}'.format(succ_succ, fail_succ))
print('pred. Fail:\t{}\t{}'.format(succ_fail, fail_fail))
