import os

from autoencoder.ApiLoader import ApiLoader
from autoencoder.EncoderData import EncoderData
from autoencoder.MovesWrapper import MovesWrapper
from env.Env import Env
from env import api
from env.mapper import state_to_matrix

batch_location = '../batches/'
expert_loader = ApiLoader(api.get_expert_list, 'Expert')
replay_loader = ApiLoader(api.get_replays_list, 'Replay')

if not os.path.exists(batch_location):
    os.makedirs(batch_location)

count_error_moves = 0
count_succ_moves = 0
count_goal_moves = 0
max_moves = 100

while count_error_moves < max_moves:
    game = replay_loader.get_next()

    if game is None:
        continue

    wrapper = MovesWrapper(game)
    while count_error_moves < max_moves and wrapper.has_next():
        state, trans = wrapper.get_next()
        data = EncoderData(state_to_matrix(state, state.dims), Env.map_action(trans.action),
                           state_to_matrix(trans.state, trans.state.dims), Env.map_reward(trans.reward))

        print(str(trans.reward) + ', ' + str(data.reward) + ', ' + str(trans.success) + ', done:' + str(trans.done))
        count_error_moves += 1
