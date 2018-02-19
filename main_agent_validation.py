import sys

from support.integrated_server import start_server
from support.last_id_store import IdStore
import agent_validation_functions
import numpy as np

max_episode_length = 100
mcts_budget = 100
max_plays = 1000
model_path = './model'

supervised_data_path = '../data_rollout/5_rollouts.pkl.zip'
# supervised_data_path = 'validate_rollout/5_rollouts.pkl.zip'
terminal_data_path = '../data_terminal/goal.pkl.zip'

id_store_supereasy_id = '04b416ab02717de528d83b9f27186e37'

use_mcts = False
use_bfs = False

run_permutations = True
run_supereasy = True
run_simple = True
run_supervised_pred = True
run_terminal_pred = True

use_integrated_server = True

if use_integrated_server:
    if not start_server():
        print('Kill process because server did not start')
        sys.exit(1)


def run_playouts(difficulty, id_store_init):
    store = IdStore(name=difficulty)
    store.write_id(id_store_init)

    if use_mcts:
        count, completed_steps, steps = agent_validation_functions.test_levels_mcts(difficulty,
                                                                                    play_length=max_episode_length,
                                                                                    model_path=model_path,
                                                                                    max_tests=max_plays,
                                                                                    id_store=store,
                                                                                    budget=mcts_budget)

        suc_count = 'N/A'
        fail_count = 'N/A'
    elif use_bfs:
        count, completed_steps, steps, frontier_lengths, explored_lengths = agent_validation_functions.test_levels_bfs(
            difficulty, play_length=max_episode_length, max_tests=max_plays, id_store=store)

        print('Frontier length average: {}, Explored length average: {}'.format(np.mean(frontier_lengths),
                                                                                np.mean(explored_lengths)))
        suc_count = 'N/A'
        fail_count = 'N/A'
    else:
        count, completed_steps, steps, suc_count, fail_count = agent_validation_functions.test_levels(difficulty,
                                                                                                      play_length=max_episode_length,
                                                                                                      model_path=model_path,
                                                                                                      max_tests=max_plays,
                                                                                                      id_store=store)

    mean_completed = 0.0 if not any(completed_steps) else np.mean(completed_steps)
    print(
        'Total: {}, Completed: {}, average steps: {}, avg steps for completed: {}, success count: {}, failure count: {}, diff: {}'.format(
            max_plays,
            count,
            np.mean(steps),
            mean_completed,
            suc_count,
            fail_count,
            difficulty))


if run_supereasy:
    print('supereasy run')
    run_playouts('supereasy', id_store_supereasy_id)

if run_permutations:
    print('permutations run')
    run_playouts('permutations-validation', '0')

if run_simple:
    print('simple run')
    run_playouts('simple-validation', '0')

if run_supervised_pred:
    print('supervised rollouts run')
    total, correct = agent_validation_functions.predict_supervised(supervised_data_path, model_path, use_mcts=use_mcts,
                                                                   mcts_budget=mcts_budget)
    acc = float(correct) / float(total) * 100.0

    print('Total: {}, correct guesses: {}, acc: {} - for Predict Supervised'.format(total, correct, acc))

if run_terminal_pred:
    print('terminal run')
    total, correct = agent_validation_functions.predict_terminal(terminal_data_path, model_path, use_mcts=use_mcts,
                                                                 mcts_budget=mcts_budget)

    acc = float(correct) / float(total) * 100.0
    print('Total: {}, correct guesses: {}, acc: {} - for Predict Supervised'.format(total, correct, acc))
