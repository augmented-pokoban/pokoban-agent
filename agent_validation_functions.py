from random import shuffle

from Network import Network
from autoencoder.EncoderData import load_object, batch_to_lists
from env.Env import Env
from env import api
from env.mapper import apply_action
from helper import process_frame
import numpy as np
import tensorflow as tf
import resource

from mcts.mcts import MCTS
from mcts.network_wrapper import NetworkWrapper
from support.BfsNode import BfsNode
import env.NewMatrixIndex as INDEX

gamma = .99  # discount rate for advantage estimation and reward discounting
height = 20
width = 20
depth = 1
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
actions = [x for x in range(a_size)]
action_failure = 0
action_success = 1


def test_levels(difficulty, play_length, model_path, max_tests, id_store):
    api.map_difficulty = difficulty
    tf.reset_default_graph()
    local_network = Network(height, width, depth, s_size, a_size, 'global', None)
    saver = tf.train.Saver(max_to_keep=5)
    print('Network initialized for {}'.format(difficulty))

    test_env = Env(id_store=id_store)

    rnn_state = local_network.state_init

    # Data collection
    completed_count = 0
    steps = []
    completed_steps = []
    success_count = 0
    failure_count = 0
    compl_factors = []
    action_results = np.zeros((a_size, 2))  # two outputs for each action

    with tf.Session() as sess:
        if model_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            print('Loading Model for {}...'.format(difficulty))
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for episode in range(max_tests):

            done = False
            s = test_env.reset()
            t = 0
            while not done and t < play_length:
                s = process_frame(s, s_size)
                a, v, rnn_state = eval_fn(sess, s, rnn_state, local_network)

                s, r, done, success = test_env.step(a)

                t += 1

                if success:
                    success_count += 1
                    action_results[a, action_success] += 1
                else:
                    failure_count += 1
                    action_results[a, action_failure] += 1

            compl_factors.append(completion_factor(s))

            # Store data
            if done:
                completed_count += 1
                completed_steps.append(t)

            steps.append(t)

    test_env.terminate()

    return completed_count, completed_steps, steps, success_count, failure_count, compl_factors, action_results


def completion_factor(state):
    count_goals = 0
    count_solved_goals = 0
    for row in range(20):
        for col in range(20):
            content = state[row, col]

            if content == INDEX.BoxAAtGoalA:
                count_solved_goals += 1
                count_goals += 1

            elif content == INDEX.AgentAtGoalA or content == INDEX.GoalA:
                count_goals += 1

    return float(count_solved_goals) / float(count_goals)


def test_levels_mcts(difficulty, play_length, model_path, max_tests, id_store, budget):
    api.map_difficulty = difficulty
    tf.reset_default_graph()
    local_network = Network(height, width, depth, s_size, a_size, 'global', None)
    saver = tf.train.Saver(max_to_keep=5)
    print('Network initialized for {}'.format(difficulty))

    test_env = Env(id_store=id_store)

    # Data collection
    completed_count = 0
    steps = []
    completed_steps = []

    with tf.Session() as sess:
        print('Loading Model for {}...'.format(difficulty))
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        for episode in range(max_tests):

            if episode % 50 == 0:
                print('Episode: {} for difficulty: {}'.format(episode, difficulty))

            rnn_state = local_network.state_init
            done = False
            s = test_env.reset()
            s = process_frame(s, s_size)
            mcts = MCTS(s, 0, NetworkWrapper(sess, rnn_state, eval_fn, local_network), s_size, False,
                        worker_name=difficulty)
            t = 0
            try:

                while not done and t < play_length:
                    _, a = mcts.search(budget)

                    s, r, done, success = test_env.step(a)
                    s = process_frame(s, s_size)
                    t += 1

                print('Episode completed: {}, memory used (kB): {}'.format(episode, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            except:
                steps.append(play_length)
                print('Failed evaluation: root equal to prev state: {}'.format(np.array_equal(process_frame(s, s_size), mcts.root.state)))
                print('Environment done: {}, mcts root done: {}'.format(done, mcts.root.done))

            # Store data
            if done:
                completed_count += 1
                completed_steps.append(t)

            steps.append(t)

    test_env.terminate()

    return completed_count, completed_steps, steps


def test_levels_bfs(difficulty, play_length, max_tests, id_store):
    api.map_difficulty = difficulty
    test_env = Env(id_store=id_store)

    # Data collection
    completed_count = 0
    steps = []
    completed_steps = []
    frontier_lengths = []
    exploration_lengths = []

    for episode in range(max_tests):

        if episode % 50 == 0:
            print('Episode: {} for difficulty: {}'.format(episode, difficulty))

        frontier = []
        frontier_set = set()
        explored = set()

        s = test_env.reset()
        done = False
        t = 0
        root = BfsNode(s, 0)
        frontier.append(root)
        frontier_set.add(root.get_hash())

        while not done and t < play_length and any(frontier):

            node = frontier.pop(0)
            frontier_set.remove(node.get_hash())

            t = node.depth

            explored.add(node.get_hash())

            # shuffle actions such that the search is not biased
            shuffle(actions)

            for a in actions:
                s, success, done = apply_action(node.state, a, Env.get_action_meanings(), reshape=False)
                next_node = BfsNode(s, node.depth + 1)

                if done:
                    t = next_node.depth
                    break

                if success and next_node.get_hash() not in explored and next_node.get_hash() not in frontier_set:
                    frontier.append(next_node)
                    frontier_set.add(next_node.get_hash())

        if done:
            completed_count += 1
            completed_steps.append(t)

        steps.append(t)
        frontier_lengths.append(len(frontier))
        exploration_lengths.append(len(explored))

    return completed_count, completed_steps, steps, frontier_lengths, exploration_lengths


def predict_supervised(data_path, model_path, use_mcts=False, mcts_budget=0):
    data = np.asarray(load_object(data_path))

    tf.reset_default_graph()
    local_network = Network(height, width, depth, s_size, a_size, 'global', None)
    saver = tf.train.Saver(max_to_keep=5)

    # data collection
    correct_predictions = 0
    total_predictions = 0

    with tf.Session() as sess:
        if model_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            print('Loading Model for Predict Supervised...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for episode in range(1000):

            if episode % 50 == 0:
                print('Processing episode {}'.format(episode))

            rnn_state = local_network.state_init

            x_state, x_action, exp_state, exp_reward, exp_success = batch_to_lists(data[episode, :], s_size)

            if use_mcts:
                mcts = MCTS(x_state[0], 0, NetworkWrapper(sess, rnn_state, eval_fn, local_network), s_size, False,
                            worker_name='predict_supervised')
            else:
                mcts = None

            correct_guess = True

            for rollout in range(5):

                if use_mcts:
                    if not correct_guess:
                        # Reset MCTS if wrong prediction in last rollout
                        mcts = MCTS(x_state[rollout], 0, NetworkWrapper(sess, rnn_state, eval_fn, local_network),
                                    s_size, False,
                                    worker_name='predict_supervised')

                    # Do mcts - must redo MCTS if wrong action is selected
                    _, a = mcts.search(mcts_budget)
                else:
                    # do network eval
                    a, _, rnn_state = eval_fn(sess, x_state[rollout], rnn_state, local_network, deterministic=True)

                correct_guess = a == x_action[rollout, 0]

                # Compare output with expected, tract prediction accuracy
                if correct_guess:
                    correct_predictions += 1

                total_predictions += 1

        return total_predictions, correct_predictions


def predict_terminal(data_path, model_path, use_mcts=False, mcts_budget=0):
    data = np.asarray(load_object(data_path))

    tf.reset_default_graph()
    local_network = Network(height, width, depth, s_size, a_size, 'global', None)
    saver = tf.train.Saver(max_to_keep=5)

    # data collection
    correct_predictions = 0
    total_predictions = 0

    with tf.Session() as sess:
        if model_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            print('Loading Model for Predict Terminal...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        x_state, x_action, exp_state, exp_reward, exp_success = batch_to_lists(data, s_size)

        for episode in range(1000):

            if episode % 50 == 0:
                print('Processing episode {}'.format(episode))

            rnn_state = local_network.state_init

            if use_mcts:
                mcts = MCTS(x_state[episode], 0, NetworkWrapper(sess, rnn_state, eval_fn, local_network), s_size, False,
                            worker_name='predict_terminal')
            else:
                mcts = None

            if use_mcts:
                # Do mcts
                _, a = mcts.search(mcts_budget)
            else:
                # do network eval
                a, _, rnn_state = eval_fn(sess, x_state[episode], rnn_state, local_network, deterministic=True)

            correct_guess = a == x_action[episode, 0]

            # Compare output with expected, tract prediction accuracy
            if correct_guess:
                correct_predictions += 1

            total_predictions += 1

        return total_predictions, correct_predictions


def eval_fn(sess, state, rnn_state, network, get_all_actions=False, deterministic=False):
    a_dist, v, rnn_state = sess.run(
        [network.policy, network.value, network.state_out],
        feed_dict={network.inputs: [state],
                   network.state_in[0]: rnn_state[0],
                   network.state_in[1]: rnn_state[1]})

    # Select the action using the prop distribution given in a_dist from previously
    if np.isnan(a_dist[0]).any():
        print(a_dist[0])

    v = v[0, 0]

    if get_all_actions:
        return a_dist[0], v, rnn_state

    if deterministic:
        a = np.argmax(a_dist[0])
    else:
        a = np.random.choice(actions, p=a_dist[0])

    return a, v, rnn_state
