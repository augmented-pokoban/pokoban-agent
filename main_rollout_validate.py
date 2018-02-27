from autoencoder.EncoderData import DataLoader, batch_to_lists, load_object, save_object
from autoencoder.EncoderNetwork import EncoderNetwork
from env.Env import Env
import tensorflow as tf
import numpy as np
from env import NewMatrixIndex

batch_size = 1000
height = 20
width = 20
depth = 1
s_size = height * width * depth
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
f_size = 7
rollout_length = 5
data_path = './validate_rollout'
model_path = './encoder_model_v2'

data = np.asarray(load_object('{}/{}_rollouts.pkl.zip'.format(data_path, rollout_length)))

count_agent = [0] * rollout_length
pred_agent_indices = []
mse_per_step = np.zeros(rollout_length, dtype=float)
errors = np.zeros((rollout_length, 1000), dtype=int)
agent_errors = np.zeros((rollout_length, 1000), dtype=int)
mse = []

tf.reset_default_graph()
trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, f_size, batch_size, 'global', trainer)
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    y_state = None

    for rollout in range(rollout_length):
        pred_agent_indices.append([])
        print('Performing rollout {}'.format(rollout + 1))
        x_state, x_action, exp_state, exp_reward, exp_success = batch_to_lists(data[:, rollout], s_size)

        if not rollout == 0:
            # hashtag ignore
            x_state = y_state

        y_reward, y_state, y_mse = network.eval_test(x_state, x_action, exp_state, sess)

        mse.append(y_mse)

        for index in range(batch_size):
            # difference count
            diff = np.absolute(exp_state[index] - y_state[index])
            errors[rollout, index] = np.sum(np.where(diff == 0, diff, 1))

            # Validate agent position: count and index
            for entry in range(400):
                if y_state[index, entry] == NewMatrixIndex.AgentAtGoalA or y_state[index, entry] == NewMatrixIndex.Agent:
                    # Test if the agent is at the same field in y_state
                    if exp_state[index, entry] == NewMatrixIndex.AgentAtGoalA or exp_state[index, entry] == NewMatrixIndex.Agent:
                        pred_agent_indices[-1].append(index)
                    else:
                        agent_errors[rollout, index] += 1

    # Completed validation
    # Output agent predictions
    print('Agent prediction counts:')
    print(count_agent)
    print()
    print()
    print('Mean errors per step:')
    print(np.mean(errors, dtype=float, axis=1))
    print()
    print('Agent indices:')
    for rollout in range(rollout_length):
        print(sorted(pred_agent_indices[rollout]))

    print('Saving mse and agent error data data.. and mean errors per rollout')
    # save_object(mse, '{}/mse_data.pkl'.format(data_path))
    # save_object(agent_errors, '{}/agent_error_data.pkl'.format(data_path))
    save_object(np.mean(errors, dtype=float, axis=1), '{}/mean_error_data.pkl'.format(data_path))



