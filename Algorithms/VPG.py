import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Networks.Basic_Networks import Policy_network, V_network


class VPG:#make it useful for both discrete(cartegorical actor) and continuous actor(gaussian policy)
    def __init__(self, state_dim, action_dim, max_action, min_action, actor=None, critic=None, buffer_size=1e6, learning_rate = 0.0003):

        self.actor = actor
        self.critic = critic

        self.buffer = Buffer(buffer_size)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action


        if self.actor == None:
            self.actor = Policy_network(self.state_dim, self.action_dim)

        if self.critic == None:
            self.critic1 = V_network(self.state_dim)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
