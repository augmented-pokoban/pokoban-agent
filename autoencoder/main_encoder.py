import os

import numpy as np
import tensorflow as tf
from env.Env import Env

from autoencoder.EncoderNetwork import EncoderNetwork

episode_max = 1000
batch_size = 64
gamma = .99  # discount rate for advantage estimation and reward discounting
height = 20
width = 20
depth = 8
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = 4  # number of different types of rewards we can get
load_model = False
model_path = './enc_model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
saver = tf.train.Saver(max_to_keep=5)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, 'global', trainer)

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

    while episode < episode_max:

        # get data
        x_state = []  # image batch
        x_action = []  # action batch
        y_state = []  # target state batch
        y_reward = []  # target reward batch

        feed_dict = {
            network.inputs: x_state,
            network.action: x_action,
            network.enc_target: y_state,
            network.val_target: y_reward
        }

        _, enc_loss, val_loss = sess.run(
                [network.train_op,
                 network.encoding_loss,
                 network.value],
                feed_dict=feed_dict)

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

        episode += 1


