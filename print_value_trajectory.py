from Network import Network
from Worker import Worker
from env import api
from env.Env import Env
from env.expert_moves import ExpertMoves
import tensorflow as tf
import numpy as np

from env.mapper import new_state_to_matrix
from helper import process_frame

height = 20
width = 20
depth = 1
s_size = height * width * depth
a_size = Env.get_action_count()
model_path = './model'

# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/c16f24c7-8889-4831-b83c-7eb1ff2b5a9b.zip'
# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/7d589ecc-ff43-4ced-8568-39141528202b.zip'
# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/c454e86b-c415-4959-8d10-3dcf71d960b0.zip'
# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/bccac59b-6f71-42a3-b59a-95b1a8747637.zip'
replay_file = 'https://pokobanserver.blob.core.windows.net/plays/6f40ee6a-ad65-490e-8113-71a6d79dd9ea.zip'
moves = ExpertMoves(api.get_expert_game(replay_file))

states = [moves.initial]

for trans in moves.transitions:
    states.append(trans.state)

states = states[:-1]

global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
tf.reset_default_graph()
network = Network(height, width, depth, s_size, a_size, 'global', None, 0.0)
saver = tf.train.Saver(max_to_keep=5)

values = []
absolutes = []
actions = []

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    rnn_state = network.state_init

    for state in states:
        s = process_frame(new_state_to_matrix(state, 20), s_size)
        a, v, rnn_state = network.eval_fn(sess, s, rnn_state)

        action = Env.get_action_meanings()[a]

        values.append(v)
        absolutes.append(abs(v))
        actions.append(action)

for i in range(len(values)):
    print('{}:\t{}\t{}'.format(actions[i], values[i], format(absolutes[i])))


