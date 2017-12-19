import datetime
from random import shuffle

import env.api as api
from autoencoder.ApiLoader import ApiLoader
from autoencoder.EncoderData import EncoderData, save_object
from autoencoder.MovesWrapper import MovesWrapper
from env.Env import Env
from env.mapper import state_to_matrix
from support import integrated_server

# mix the data somehow on-going
# for each transition, add to list
# # if list contains 1000 elements, save it and load new (use timestamp as name)
# Convert reward to index
# Convert action to index
# convert states

max_list_size = 1024
max_batches = 1000
expert_ratio = 0.9
replay_count = 0
expert_count = 0
total = 0
batch = []
expert_loader = ApiLoader(api.get_expert_list, 'Expert')
replay_loader = ApiLoader(api.get_replays_list, 'Replay')
integrated_server.start_server()  # running using integrated server

while (expert_loader.has_next() or replay_loader.has_next()) and total < (max_list_size * max_batches):

    if total == 0 and expert_loader.has_next():
        game = expert_loader.get_next()
        kind = 'EXPERT'
    elif expert_count / total < expert_ratio and expert_loader.has_next():
        game = expert_loader.get_next()
        kind = 'EXPERT'
    else:
        game = replay_loader.get_next()
        kind = 'REPLAY'

    wrapper = MovesWrapper(game)

    while wrapper.has_next() and total < (max_list_size * max_batches):
        state, trans = wrapper.get_next()
        data = EncoderData(state_to_matrix(state, state.dims), Env.map_action(trans.action),
                           state_to_matrix(trans.state, trans.state.dims), Env.map_reward(trans.reward))
        batch.append(data)
        total += 1
        if kind == 'EXPERT':
            expert_count += 1
        elif kind == 'REPLAY':
            replay_count += 1

        if total % max_list_size == 0:
            print('Total: {}, Experts: {}, Replays: {} - saving file...'.format(total, expert_count, replay_count))
            shuffle(batch)
            save_object(batch, 'batches/' + str(datetime.datetime.now()) + '.pkl')
            batch = []

print('Terminating with total: {} leaving {} elements processed but not stored.'.format(total, total % max_list_size))



