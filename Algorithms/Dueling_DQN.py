#Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, 2015.

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Networks.Dueling_Network import Dueling_Network

class Dueling_DQN:
    def __init__(self, state_dim, action_dim, network=None, batch_size=100, buffer_size=1e6, gamma=0.99, learning_rate=0.001, epsilon=0.2, training_start=200):
        self.network = network

        self.buffer = Buffer(buffer_size)
        self.optimizer = tf.optimizers.Adam(learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = epsilon
        self.training_start = training_start

        self.step = 0

        if self.network == None:
            self.network = Dueling_Network(self.state_dim, self.action_dim)
            print("Network made")

        self.network_list = {'Network': self.network}

    def get_action(self, state):
        state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        q_value = self.network(state).numpy()

        if np.random.random() < self.eps:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            return np.argmax(q_value, axis=1)[0]

    def train(self, training_num):
        self.step += 1

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        target_q = r + self.gamma * (1 - d) * tf.reduce_max(self.network(ns), axis=1, keepdims=True)
        target_q = tf.stop_gradient(target_q)

        with tf.GradientTape() as tape:

            q_value = tf.reduce_sum(
                self.network(s) * tf.squeeze(tf.one_hot(tf.dtypes.cast(a, tf.int32), self.action_dim), axis=1), axis=1, keepdims=True)
            loss = 0.5*tf.math.reduce_mean(tf.square(target_q - q_value))

        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))





