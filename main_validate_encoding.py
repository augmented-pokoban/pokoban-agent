from autoencoder.EncoderData import DataLoader, batch_to_lists
from autoencoder.EncoderNetwork import EncoderNetwork
from env.Env import Env
import tensorflow as tf
import numpy as np
from env import NewMatrixIndex

batch_size = 1024
height = 20
width = 20
depth = 1
s_size = height * width * depth
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
f_size = 7

data = DataLoader('data.csv', batch_size, '../batches/')

model_path = './encoder_model_v2'
agent_prediction = 0
mse = []
total = 0

tf.reset_default_graph()
trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, f_size, batch_size, 'global', trainer)
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    while data.has_val():

        # [s_size] in new matrix form for both x_state and exp_state
        x_state, x_action, exp_state, exp_reward, exp_success = batch_to_lists(data.get_val(), s_size)
        y_reward, y_state, y_success, y_mse = network.eval_test(x_state, x_action, exp_state, sess)

        for i in range(batch_size):
            # y_state is [s_size] in rounded, new matrix form

            for entry in range(400):
                if exp_state[i, entry] == NewMatrixIndex.AgentAtGoalA or exp_state[i, entry] == NewMatrixIndex.Agent:
                    # Test if the agent is at the same field in y_state
                    agent_prediction += 1 if y_state[i, entry] == NewMatrixIndex.AgentAtGoalA \
                                             or y_state[i, entry] == NewMatrixIndex.Agent else 0
                    break

        total += batch_size
        mse += y_mse

        if total % (10 * batch_size) == 0:
            print('Batches processed: {}, correct agent predictions (total): {}'.format(total / batch_size,
                                                                                        agent_prediction))

    # Completed validation
    mse = np.mean(mse)
    agent_acc = float(agent_prediction) / float(total) * 100
    print('Validation MSE: {}, agent prediction rate: {} %'.format(mse, agent_acc))
