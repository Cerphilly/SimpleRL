#Simple statistical gradient-following algorithms for connectionist reinforcement learning, Ronald J. Williams, 1992

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Networks.Basic_Networks import Policy_network


class REINFORCE:
    def __init__(self, state_dim, action_dim, network=None, gamma=0.99, learning_rate=0.001):
        self.network = network

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_action(self, state):
        pass

    def train(self):
        pass