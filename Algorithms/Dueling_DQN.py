#Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, 2015.

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Networks.Dueling_Network import Dueling_Network

class Dueling_DQN:
    def __init__(self, state_dim, action_dim, args):
        self.buffer = Buffer(args.buffer_size)
        self.optimizer = tf.optimizers.Adam(args.learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.eps = args.epsilon
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0

        self.network = Dueling_Network(self.state_dim, self.action_dim, args.hidden_dim)

        self.network_list = {'Network': self.network}
        self.name = 'Dueling DQN'

    def get_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)

        q_value = self.network(state).numpy()

        if np.random.random() < self.eps:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            return np.argmax(q_value, axis=1)[0]
        
    def eval_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)

        q_value = self.network(state).numpy()

        return np.argmax(q_value, axis=1)[0]

    def train(self, training_num):
        total_loss = 0

        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            target_q = r + self.gamma * (1 - d) * tf.reduce_max(self.network(ns), axis=1, keepdims=True)
            target_q = tf.stop_gradient(target_q)

            with tf.GradientTape() as tape:

                q_value = tf.reduce_sum(self.network(s) * tf.squeeze(tf.one_hot(tf.dtypes.cast(a, tf.int32), self.action_dim), axis=1), axis=1, keepdims=True)
                loss = 0.5 * tf.math.reduce_mean(tf.square(target_q - q_value))

            variables = self.network.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            total_loss += loss.numpy()

        return [['Loss/Loss', total_loss]]





