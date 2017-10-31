import tensorflow as tf
from Network import Network
from Worker import Worker
from env.Env import Env

max_episode_length = 301
max_buffer_length = 20
gamma = .99  # discount rate for advantage estimation and reward discounting
height = 20
width = 20
depth = 8
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = len(Env(use_server=False).get_action_meanings())  # Agent can move Left, Right, or Fire
load_model = False
level = None
model_path = './model'

tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
    master_network = Network(height, width, depth, s_size, a_size, 'global', None)  # Generate global network
    num_workers = 1  # Set workers ot number of available CPU threads

    print('Creating', num_workers, 'workers')
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(
            Worker(i, (height, width, depth, s_size), a_size, trainer, model_path, global_episodes, explore_self=True))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(1):
        workers[0].play(sess, 0, level=level)
