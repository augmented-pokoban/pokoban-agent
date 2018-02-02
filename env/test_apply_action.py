import numpy as np

from autoencoder.EncoderData import DataLoader, batch_to_lists
from env import api
from env.mapper import *
from env.Env import Env
from helper import reshape_back

batch_size = 1024
height = 20
width = 20
depth = 1
s_size = height * width * depth
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
f_size = 7

data = DataLoader('../data.csv', batch_size, '../batches/')

sample = data.get_train()
x_state, x_action, exp_state, exp_reward, exp_success = batch_to_lists(sample, s_size)

success_data = []
state_data = []
done_data = []

count = 0
for i in range(batch_size):

    state, success, done = apply_action(x_state[i], x_action[i][0], Env.get_action_meanings())

    # success_data.append(success == exp_success[i][0])
    # done_data.append(done == sample[i].done)
    equal = success == exp_success[i] #np.array_equal(exp_state[i], state)

    exp_r = exp_reward[i][0]

    if not equal:
        print('Success from data: {}, success from apply_action(): {}'.format(exp_success[i], success))
        print('Done from data: {}, done from apply_action(): {}'.format(sample[i].done, done))
        apply_action(x_state[i], x_action[i][0], Env.get_action_meanings())

        result = dict()
        result['errors'] = 0
        # result['missing_errors'] = np.sum(missing)
        # result['overfit_errors'] = np.sum(overfit)
        result['x_state'] = new_matrix_to_state(reshape_back(x_state[i], height, width), 20)
        result['y_state_exp'] = new_matrix_to_state(reshape_back(exp_state[i], height, width), 20)
        result['y_state_eval'] = new_matrix_to_state(reshape_back(state, height, width), 20)
        # result['diff'] = new_matrix_to_state(reshape_back(diff, height, width), 20)
        result['y_reward_exp'] = Env.get_reward_meanings()[exp_r]
        result['y_reward_eval'] = done
        result['action'] = Env.get_action_meanings()[x_action[i, 0]]
        result['success'] = success
        # result['missing'] = matrix_to_state(reshape_back(missing, height, width, depth), 20)
        # result['overfit'] = matrix_to_state(reshape_back(overfit, height, width, depth), 20)

        api.post_encoding(result)
        print('Submitted {}'.format(count + 1))
        count += 1


    # if not (state_data[-1] and success_data[-1] and done_data[-1]):
    #     x_state_re = helper.reshape_back(x_state[i], 20, 20)
    #     act_state = helper.reshape_back(state, 20, 20)
    #     exp_state = helper.reshape_back(exp_state[i], 20, 20)
    #     print('Success equal: {}'.format(success_data[-1]))
    #     print('Done equal: {}'.format(done_data[-1]))
    #     print('State equal: {}'.format(state_data[-1]))
    #     print('Action: {}'.format(Env.get_action_meanings()[x_action[i][0]]))
    #     print('X_State:')
    #     print(x_state_re)
    #     print('Expected state:')
    #     print(exp_state)
    #     print('Actual state:')
    #     print(act_state)
    #     print('Diff:')
    #     print(exp_state - act_state)
    #     print('====' * 20)

print(state_data)
