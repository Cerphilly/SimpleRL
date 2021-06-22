#Deep Reinforcement Learning with Double Q-learning, Hasselt et al 2015. Algorithm: Double DQN

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight
from Networks.Basic_Networks import Policy_network


class DDQN:
    def __init__(self, state_dim, action_dim, args):

        self.buffer = Buffer(args.buffer_size)

        self.optimizer = tf.keras.optimizers.Adam(args.learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.epsilon = args.epsilon
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0
        self.copy_iter = args.copy_iter

        self.network = Policy_network(self.state_dim, self.action_dim, args.hidden_dim)
        self.target_network = Policy_network(self.state_dim, self.action_dim, args.hidden_dim)

        copy_weight(self.network, self.target_network)

        self.network_list = {'Network': self.network, 'Target_Network': self.target_network}
        self.name = 'Double DQN'

    def get_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)

        q_value = self.network(state, activation='linear').numpy()
        best_action = np.argmax(q_value, axis=1)[0]

        if np.random.random() < self.epsilon:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            return best_action

    def eval_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)

        q_value = self.network(state, activation='linear').numpy()
        best_action = np.argmax(q_value, axis=1)[0]

        return best_action

    def train(self, training_num):
        total_loss = 0

        for i in range(training_num):
            self.current_step += 1
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


            if self.current_step % self.copy_iter == 0:
                copy_weight(self.network, self.target_network)

            total_loss += loss.numpy()

        return [['Loss/Loss', total_loss]]


