import os

import tensorflow as tf
from Network import Network
from autoencoder.EncoderData import save_object
from env import api
from generate_random_plays_worker import Worker
from env.Env import Env

gamma = .99  # discount rate for advantage estimation and reward discounting
height = 20
width = 20
depth = 1
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = len(Env.get_action_meanings())  # Agent can move Left, Right, or Fire
load_model = False
num_workers = 1
model_path = './play_model'
data_path = './succ_fail_plays'
api.map_difficulty = 'medium'  # remember to set unsupervised api to supervised

tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
    master_network = Network(height, width, depth, s_size, a_size, 'global', None)  # Generate global network

    print('Creating', num_workers, 'workers')

    worker = Worker(0, (height, width, depth, s_size), a_size, trainer, model_path, global_episodes, explore_self=True)
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    sess.run(tf.global_variables_initializer())

    failures, successes = worker.play(sess)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    save_object(failures, '{}/failures.pkl'.format(data_path))
    save_object(successes, '{}/successes.pkl'.format(data_path))

