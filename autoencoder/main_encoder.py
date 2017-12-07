import os
import numpy as np
import tensorflow as tf
from autoencoder.EncoderData import DataReader
from env.Env import Env
from autoencoder.EncoderNetwork import EncoderNetwork
import helper

episode_max = 1000
batch_size = 32
gamma = .99  # discount rate for advantage estimation and reward discounting
height = 20
width = 20
depth = 8
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
# 0 = -1
# 1 = -0.1
# 2 = 1
# 3 = 10
load_model = False
model_path = './enc_model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, batch_size, 'global', trainer)
saver = tf.train.Saver(max_to_keep=5)
data = DataReader()

with tf.Session() as sess:
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # Train train train
    # Chuu chuu
    summary_writer = tf.summary.FileWriter("Autoencoder")

    episode = 0

    episode_enc_loss = []
    episode_val_loss = []

    while episode < episode_max and data.has_more(batch_size):
        episode += 1
        batch = data.get_batch(batch_size)
        # get data
        x_state = np.array(list(map(lambda encoded_data: helper.process_frame(encoded_data.state_x, s_size), batch)))  # image biartch
        x_action = np.array(list(map(lambda encoded_data: [encoded_data.action], batch)))  # action batch
        y_state = np.array(list(map(lambda encoded_data: helper.process_frame(encoded_data.state_y, s_size), batch)))  # target state batch
        y_reward = np.array(list(map(lambda encoded_data: [encoded_data.reward], batch)))  # target reward batch

        feed_dict = {
            network.input_image: x_state,
            network.action: x_action,
            network.enc_target: y_state,
            network.reward: y_reward
        }

        _, enc_loss, val_loss = sess.run(
            [
                network.train_op,
                network.encoding_loss,
                network.value_loss
            ],
            feed_dict=feed_dict
        )

        episode_enc_loss.append(enc_loss)
        episode_val_loss.append(val_loss)

        if episode % 5 == 0:
            mean_enc_loss = np.mean(episode_enc_loss[-5:])
            mean_val_loss = np.mean(episode_val_loss[-5:])
            summary = tf.Summary()
            summary.value.add(tag="Encoding Loss", simple_value=float(mean_enc_loss))
            summary.value.add(tag="Reward Loss", simple_value=float(mean_val_loss))
            summary_writer.add_summary(summary, episode)
            summary_writer.flush()

        if episode % 1000 == 0:
            saver.save(sess, model_path + '/model-' + str(episode) + '.cptk')

    print('Episodes: {}'.format(episode))


