#Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, remove_argument
from Network.Basic_Network import Policy_network

class DQN:
    def __init__(self, state_dim, action_dim, args):

        self.buffer = Buffer(state_dim, 1, args.buffer_size)
        self.optimizer = tf.keras.optimizers.Adam(args.learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0
        self.copy_iter = args.copy_iter

        self.network = Policy_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.target_network = Policy_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)

        copy_weight(self.network, self.target_network)

        self.network_list = {'Network': self.network, 'Target_Network': self.target_network}
        self.name = 'DQN'

    @staticmethod
    def get_config(parser):
        parser.add_argument('--epsilon', default=0.1, type=float, help='Action Exploration probability')
        parser.add_argument('--copy-iter', default=100, type=int, help='Frequency to update target network')
        remove_argument(parser, ['actor_lr', 'critic_lr', 'v_lr'])

        return parser

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(low=0, high=self.action_dim)

        else:
            state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
            q_value = self.network(state).numpy()
            best_action = np.argmax(q_value, axis=1)[0]
            return best_action

    def eval_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        q_value = self.network(state).numpy()
        best_action = np.argmax(q_value, axis=1)[0]

        return best_action


    def train(self, training_num):
        total_loss = 0
        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            target_q = tf.reduce_max(self.target_network(ns), axis=1, keepdims=True)
            target_value = r + self.gamma * (1 - d) * target_q
            target_value = tf.stop_gradient(target_value)

            with tf.GradientTape() as tape:
                selected_values = tf.reduce_sum(self.network(s) * tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), self.action_dim), axis=1), axis=1, keepdims=True)
                loss = 0.5*tf.reduce_mean(tf.square(target_value - selected_values))

            variables = self.network.trainable_variables
            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

            if self.current_step % self.copy_iter == 0:
                copy_weight(self.network, self.target_network)

            total_loss += loss.numpy()

        return [['Loss/Loss', total_loss]]







