import os
from random import randint
import env.api as api
from autoencoder.ApiLoader import ApiLoader
from autoencoder.EncoderData import EncoderData, save_object
from env.Env import Env
from env.mapper import state_to_matrix

# mix the data somehow on-going
# for each transition, add to list
# # if list contains 1000 elements, save it and load new (use timestamp as name)
# Convert reward to index
# Convert action to index
# convert states

total = 0
batch = 1000
rollout_length = 5
expert_loader = ApiLoader(api.get_expert_list, 'Expert')
batch_location = 'validate_rollout/'

if not os.path.exists(batch_location):
    os.makedirs(batch_location)

rollouts = []

for i in range(batch):
    game = []

    if total % 50 == 0:
        print('Total: {}'.format(total))

    while expert_loader.has_next():
        state, trans = expert_loader.get_next()
        data = EncoderData(state_to_matrix(state, state.dims),
                           Env.map_action(trans.action),
                           state_to_matrix(trans.state, trans.state.dims),
                           Env.map_reward(trans.reward),
                           trans.success,
                           trans.done)

        game.append(data)

        if trans.done and len(game) < rollout_length:
            # Get next game, this one does not have enough transitions
            game = []
            print('Skipping game..')
            continue

        if trans.done:

            # first start indicies at random:
            start_index = randint(0, len(game) - rollout_length)

            rollout = game[start_index:start_index+5]

            # Do the magic here
            rollouts.append(rollout)

            total += 1
            break

save_object(rollouts, batch_location + '{}_rollouts.pkl'.format(rollout_length))



