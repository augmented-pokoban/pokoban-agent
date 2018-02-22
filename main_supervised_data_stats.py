import numpy as np

# Retrieve supervised datasets with initial store
# Retrieve individual trajectories
# Aggregate data for each action
# repeat until 140,000 has been retrieved
import sys

from env.Env import Env
from support.integrated_server import start_server
from support.last_id_store import IdStore

max_episodes = 140000
print_every = 500
use_integrated_server = True

# Startup the server for fun and glory
if use_integrated_server:
    if not start_server():
        print('Kill process because server did not start')
        sys.exit(1)

actions = np.zeros(8)
env = Env(use_server=False, id_store=IdStore('supervised_statistics'))
episode = 0

while episode < max_episodes:

    if episode % print_every == 0:
        print('Episodes processed: {}'.format(episode))

    done = False
    _ = env.reset()

    while not done:
        a, v = env.get_expert_action_value()
        s, r, done, success = env.step()

        actions[a] += 1

    episode += 1

    if episode % 5000 == 0:
        summed = np.sum(actions)
        for i in range(8):
            print('{}: \t{} \t {}%'.format(Env.get_action_meanings()[i], actions[i],
                                           float(actions[i]) / float(summed) * 100.0))

        print()



