from autoencoder.EncoderData import DataLoader
from env.Env import Env

batch_path = '../batches/'
metadata_file = 'data.csv'
action_dist = [0] * len(Env.get_action_meanings())
reward_dist = [0] * len(Env.get_reward_meanings())
success_dist = [0] * 2
count = 0
count_batches = 0

data = DataLoader(metadata_file, 1024, batch_path, 0)

while True:
    train_set = data.get_val()

    if train_set is None:
        break

    count_batches += 1

    for trans in train_set:

        trans.reward = trans.reward if not trans.done else 3
        count += 1
        action_dist[trans.action] += 1
        reward_dist[trans.reward] += 1

        succ = 1 if trans.success else 0

        success_dist[succ] += 1

    if count_batches % 10 == 0:
        print('Processed batches: {}'.format(count_batches))

print('Total batches: {}, transitions: {}'.format(count_batches, count))
print()
print('Actions:')

for index, action in enumerate(Env.get_action_meanings()):
    print('\t{}:\t{}'.format(action, action_dist[index]))

print()
print('Rewards:')
for index, reward in enumerate(Env.get_reward_meanings()):
    print('\t{}:\t{}'.format(reward, reward_dist[index]))

print()
print('Game success:')
print('\tFailure:\t{}'.format(success_dist[0]))
print('\tSuccess:\t{}'.format(success_dist[1]))




