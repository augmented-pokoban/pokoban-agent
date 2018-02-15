import os
from random import shuffle

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
max_of_each = 1000
expert_loader = ApiLoader(api.get_expert_list, 'Expert')
batch_location = 'validate_terminal_state/'

if not os.path.exists(batch_location):
    os.makedirs(batch_location)

close_to_goal_indices = [-2, -3]

close_to_goal = []
goal = []

for i in range(max_of_each):
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

        if trans.done and len(game) < 3:
            # Get next game, this one does not have enough transitions
            game = []
            continue

        if trans.done:
            # Do the magic here
            goal.append(game[-1])

            # select -2 or -3
            shuffle(close_to_goal_indices)
            close_to_goal.append(game[close_to_goal_indices[0]])
            total += 1
            break

save_object(close_to_goal, batch_location + 'close_to_goal.pkl')
save_object(goal, batch_location + 'goal.pkl')



