#Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight
from Networks.Basic_Networks import Policy_network

class DQN:
    def __init__(self, state_dim, action_dim, training_step=100, batch_size=100, buffer_size=1e6, gamma=0.99, learning_rate=0.001, epsilon=0.1, training_start=200, copy_iter=5):

        self.buffer = Buffer(buffer_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.training_start = training_start
        self.training_step = training_step
        self.copy_iter = copy_iter


        self.network = Policy_network(self.state_dim, self.action_dim)
        self.target_network = Policy_network(self.state_dim, self.action_dim)

        copy_weight(self.network, self.target_network)

        self.network_list = {'Network': self.network, 'Target_Network': self.target_network}
        self.name = 'DQN'

    def get_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)
        q_value = self.network(state, activation='linear').numpy()
        best_action = np.argmax(q_value, axis=1)[0]

        if np.random.random() < self.epsilon:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            return best_action

    def train(self, training_num):

        for i in range(training_num):

            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            target_q = tf.reduce_max(self.target_network(ns, activation='linear'), axis=1, keepdims=True)
            target_value = r + self.gamma*(1-d)*target_q
            target_value = tf.stop_gradient(target_value)

            with tf.GradientTape() as tape:
                selected_values = tf.reduce_sum(self.network(s, activation='linear')*tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), self.action_dim), axis=1), axis=1, keepdims=True)
                loss = 0.5*tf.reduce_mean(tf.square(target_value - selected_values))

            variables = self.network.trainable_variables
            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))


            if i % self.copy_iter == 0:
                copy_weight(self.network, self.target_network)






