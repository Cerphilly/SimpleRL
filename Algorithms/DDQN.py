#Deep Reinforcement Learning with Double Q-learning, Hasselt et al 2015. Algorithm: Double DQN

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight
from Networks.Basic_Networks import Policy_network


class DDQN:
    def __init__(self, state_dim, action_dim, network=None, target_network=None, training_step=100, batch_size=100, buffer_size=1e6, gamma=0.99, learning_rate=0.001, epsilon=0.1, training_start=200, copy_iter=5):
        self.network = network
        self.target_network = target_network

        self.buffer = Buffer(buffer_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = learning_rate
        self.epsilon = epsilon
        self.training_start = training_start
        self.training_step = training_step
        self.copy_iter = copy_iter

        self.step = 0

        if self.network == None:
            self.network = Policy_network(self.state_dim, self.action_dim)
            print("Network made")
        if self.target_network == None:
            self.target_network = Policy_network(self.state_dim, self.action_dim)
            print("Target network made")

        copy_weight(self.network, self.target_network)

        self.network_list = {'Network': self.network, 'Target_Network': self.target_network}

    def get_action(self, state):
        state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        q_value = self.network(state, activation='linear').numpy()
        best_action = np.argmax(q_value, axis=1)[0]

        if np.random.random() < self.epsilon:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            return best_action

    def train(self, training_num):
        self.step += 1

        total_loss = 0

        for i in range(training_num):
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            q_value = tf.expand_dims(tf.argmax(self.network(ns, activation='linear'), axis=1, output_type=tf.int32), axis=1)
            q_value_one = tf.squeeze(tf.one_hot(q_value, depth=self.action_dim), axis=1)

            target_value = r + self.gamma*(1-d)*tf.reduce_sum(self.target_network(ns, activation='linear')*q_value_one, axis=1, keepdims=True)
            target_value = tf.stop_gradient(target_value)

            with tf.GradientTape() as tape:
                selected_values = tf.reduce_sum(self.network(s, activation='linear')*tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), self.action_dim), axis=1), axis=1, keepdims=True)
                loss = 0.5*tf.math.reduce_mean(tf.square(target_value - selected_values))

            variables = self.network.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            total_loss += loss.numpy()

            if self.step % self.copy_iter == 0:
                copy_weight(self.network, self.target_network)

            del tape

        return total_loss
