from Network import Network
from autoencoder.EncoderData import save_object
from env import api
from env.Env import Env
from env.expert_moves import ExpertMoves
import tensorflow as tf
import numpy as np

from env.mapper import new_state_to_matrix
from helper import process_frame, discount

height = 20
width = 20
depth = 1
s_size = height * width * depth
a_size = Env.get_action_count()
model_path = '../model'

# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/c16f24c7-8889-4831-b83c-7eb1ff2b5a9b.zip'
# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/7d589ecc-ff43-4ced-8568-39141528202b.zip'
# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/c454e86b-c415-4959-8d10-3dcf71d960b0.zip'
# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/bccac59b-6f71-42a3-b59a-95b1a8747637.zip'
# replay_file = 'https://pokobanserver.blob.core.windows.net/plays/6f40ee6a-ad65-490e-8113-71a6d79dd9ea.zip'
# moves = ExpertMoves(api.get_expert_game(replay_file))

max_episodes = 1000

last_id = None
maps = []
response = api.get_expert_list(last_id, max_episodes, 'asc')
maps += list(map(lambda expert_games: expert_games['fileRef'], response['data']))
last_id = response['data'][-1]['_id']

global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
tf.reset_default_graph()
network = Network(height, width, depth, s_size, a_size, 'global', None, 0.0)
saver = tf.train.Saver(max_to_keep=5)

values = np.zeros((1000, 21))

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode, level in enumerate(maps):

        if episode % 50 == 0:
            print('Episode: {}'.format(episode))

        moves = ExpertMoves(api.get_expert_game(level))

        states = [moves.initial]

        for trans in moves.transitions:
            states.append(trans.state)

        states = states[:-1]

        rewards = []
        for trans in moves.transitions:
            rewards.append(trans.reward)

        rnn_state = network.state_init

        for step, state in enumerate(states):
            s = process_frame(new_state_to_matrix(state, 20), s_size)
            a, v, rnn_state = network.eval_fn(sess, s, rnn_state)

            action = Env.get_action_meanings()[a]

            values[episode, step] = v

        last_s = moves.transitions[-1].state

        s = process_frame(new_state_to_matrix(last_s, 20), s_size)
        a, v, rnn_state = network.eval_fn(sess, s, rnn_state)

        values[episode, 20] = v

        # Calculate advantages
        # rewards_plus = np.asarray(rewards + [0.0])
        # discounted_rewards = discount(rewards_plus, 0.99)[:-1]
        #
        # value_plus = np.asarray(values[0, :].tolist() + [0.0])
        #
        # advantages = rewards + 0.99 * value_plus[1:] - value_plus[:-1]
        # advantages_disc = discount(advantages, 0.99)
        #
        # value_diff = value_plus[1:] - value_plus[:-1]
        #
        # data = {
        #     'value_diff': value_diff,
        #     'rewards': discounted_rewards,
        #     'values': values.tolist(),
        #     'adv_disc': advantages_disc
        # }

        # save_object(data, 'adv_r_v.pkl')

means = np.mean(values, axis=0)

save_object(means, 'values_21_steps.pkl')

