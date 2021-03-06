import sys

import numpy as np

from Network import *
from env.Env import Env
from helper import update_target_graph, discount, process_frame
from mcts.mcts import MCTS
from mcts.network_wrapper import NetworkWrapper
from support.last_id_store import IdStore


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
        #self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        #discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {#self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([#self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        return p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver, max_buffer_length):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        total_levels = 0
        print("Starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():

            # This is the beginning of an episode
            while not coord.should_stop() and self.env.has_more_data():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                done = False

                s = self.env.reset()
                total_levels += 1
                s = process_frame(s, self.s_size)

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                mcts = None
                if self.use_mcts:
                    mcts = MCTS(s, 50, self.env, NetworkWrapper(sess, rnn_state, self.eval_fn))

                while not done and episode_step_count < max_episode_length:
                    if self.use_mcts:
                        if not self.explore_self:
                            print('The worker should be set to explore self when using MCTS. Terminating...')
                            sys.exit(1)

                        a = mcts.search(self.searches)
                        v = self.value_fn(sess, s, rnn_state)
                    elif self.explore_self:
                        a, v, rnn_state = self.eval_fn(sess, s, rnn_state)
                    else:
                        a, v = self.env.get_expert_action_value()

                    # Create step
                    try:
                        s1, r, done, _ = self.env.step(a)
                    except Exception as e:
                        self.env._store = True
                        self.env.terminate('episode count: ' + str(episode_count))
                        print(e)
                        sys.exit()

                    if done:
                        print('Episode: {} Steps: {} Worker: {} - Done: True'.format(episode_count, episode_step_count,
                                                                                     self.name))

                    # Update values, states, total amount of steps, etc
                    episode_buffer.append([s, a, r, s1, done, v])
                    episode_values.append(v)

                    episode_reward += r
                    s = process_frame(s1, self.s_size)
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == max_buffer_length and not done \
                            and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = self.value_fn(sess, s, rnn_state)
                        # Train here
                        p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) is not 0:
                    p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                if episode_count % 100 == 0 and self.name == 'worker_0' and episode_count is not 0:
                    print('Saved model')
                    self.save(saver, sess, episode_count)
                    self.play(sess, episode_count)

                # Periodically save model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count is not 0:
                    if self.name == 'worker_0':
                        print('Episode:', episode_count, 'steps: ', episode_step_count)

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    summary.value.add(tag='Levels', simple_value=float(total_levels))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)

                episode_count += 1
                sys.stdout.flush()

            print(self.name, 'completed training in episode', str(episode_count))
            sys.stdout.flush()
            self.save(saver, sess, episode_count)

    def play(self, sess, episode_count, level=None):

        play_env = self.env.get_play_env()

        done = False
        s = play_env.reset(store=True, level=level)
        s = process_frame(s, self.s_size)

        rnn_state = self.local_AC.state_init
        self.batch_rnn_state = rnn_state

        t = 0
        success = True

        while not done and t < 300:
            if not success:
                a, v, rnn_state = self.eval_fn(sess, s, rnn_state, deterministic=False)
            else:
                a, v, rnn_state = self.eval_fn(sess, s, rnn_state, deterministic=True)

            s, r, done, success = play_env.step(a)
            s = process_frame(s, self.s_size)
            t += 1

        play_env.terminate('episode count: ' + str(episode_count))
        print('Test trial terminated')

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
