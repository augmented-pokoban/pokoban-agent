import sys
from random import random

import numpy as np
from Network import *
from autoencoder.EncoderData import EncoderData
from env.Env import Env
from helper import update_target_graph, discount, process_frame
from support.last_id_store import IdStore
from support.stats_object import StatsObject


class Worker:
    def __init__(self, name, dimensions, a_size, trainer, model_path, global_episodes, explore_self=True,
                 use_mcts=False, searches=10):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.explore_self = explore_self
        self.use_mcts = use_mcts
        self.searches = searches

        self.height, self.width, depth, self.s_size = dimensions

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = Network(self.height, self.width, depth, self.s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        # End A3C basic setup
        self.actions = range(a_size)
        self.env = Env(explore_self, id_store=IdStore(self.name))

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        # next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        # self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        # discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {  # self.local_AC.target_v: discounted_rewards,
            self.local_AC.inputs: np.vstack(observations),
            self.local_AC.actions: actions,
            self.local_AC.advantages: advantages,
            self.local_AC.state_in[0]: self.batch_rnn_state[0],
            self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([  # self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def play(self, sess):

        episode = 0
        data_success = []
        data_failure = []
        count_success = 0
        count_failure = 0

        # Should be normal env initialized from a high level number
        play_env = self.env.get_play_env()
        prob_default = 1.0 / 25.0
        prob = prob_default

        while count_success < 1000 or count_failure < 1000:

            done = False
            s = play_env.reset()
            s = process_frame(s, self.s_size)

            rnn_state = self.local_AC.state_init
            self.batch_rnn_state = rnn_state

            t = 0

            while not done and t < 50 and (count_success < 1000 or count_failure < 1000):
                a, v, rnn_state = self.eval_fn(sess, s, rnn_state, deterministic=False)

                s1, r, done, success = play_env.step(a)

                s1 = process_frame(s1, self.s_size)

                # Store data here with some probability
                if random() < prob:
                    if success and count_success < 1000:
                        # Do the success
                        data = EncoderData(s, a, s1, Env.map_reward(r), success, done)
                        data_success.append(data)
                        count_success += 1
                        prob = prob_default

                    elif not success and count_failure < 1000:
                        # Do this
                        data = EncoderData(s, a, s1, Env.map_reward(r), success, done)
                        data_failure.append(data)
                        count_failure += 1
                        prob = prob_default
                    else:
                        # if succ is full and success, take next failure
                        # if failure is full and not success, take next success
                        prob = 1.0

                s = s1
                t += 1

            episode += 1

            if episode % 10 == 0:
                print('Episode: {}, Failures: {}, Successes: {}'.format(episode, count_failure, count_success))

        play_env.terminate()
        print('Test trial terminated')
        return data_failure, data_success

    def save(self, saver, sess, episode_count):
        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')

    def eval_fn(self, sess, state, rnn_state, deterministic=False):

        a_dist, v, rnn_state = sess.run(
            [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
            feed_dict={self.local_AC.inputs: [state],
                       self.local_AC.state_in[0]: rnn_state[0],
                       self.local_AC.state_in[1]: rnn_state[1]})

        # Select the action using the prop distribution given in a_dist from previously
        if np.isnan(a_dist[0]).any():
            print(a_dist[0])

        a, v = None, v[0, 0]

        if deterministic:
            a = np.argmax(a_dist)
        else:
            a = np.random.choice(self.actions)

        return a, v, rnn_state

    def value_fn(self, sess, state, rnn_state):
        return sess.run(self.local_AC.value,
                        feed_dict={self.local_AC.inputs: [state],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
