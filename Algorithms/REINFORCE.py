#Simple statistical gradient-following algorithms for connectionist reinforcement learning, Ronald J. Williams, 1992

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Networks.Basic_Networks import Policy_network
from Networks.Gaussian_Actor import Gaussian_Actor


class REINFORCE:
    def __init__(self, state_dim, action_dim, discrete, hidden_dim=256, training_step=1, gamma=0.99, learning_rate=0.001):

        self.buffer = Buffer()

        self.state_dim = state_dim
        self.action_dim = action_dim

        
        self.discrete = discrete

        self.gamma = gamma
        self.training_start = 0
        self.training_step = training_step
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        if discrete == True:
            self.network = Policy_network(self.state_dim, self.action_dim, (hidden_dim, hidden_dim))
        else:
            self.network = Gaussian_Actor(self.state_dim, self.action_dim, (hidden_dim, hidden_dim))

        self.network_list = {'Network': self.network}
        self.name = 'REINFORCE'


    def get_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)

        if self.discrete == True:
            policy = self.network(state, activation='softmax').numpy()[0]
            action = np.random.choice(self.action_dim, 1, p=policy)[0]
            
        else:
            action = self.network(state).numpy()[0]
            action = np.clip(action, -1, 1)

        return action


    def train(self, training_num):
        s, a, r, ns, d = self.buffer.all_sample()
        returns = np.zeros_like(r.numpy())

        running_return = 0
        for t in reversed(range(len(r))):
            running_return = r[t] + self.gamma * running_return * (1-d[t])
            returns[t] = running_return


        with tf.GradientTape() as tape:
            if self.discrete == True:
                policy = self.network(s, activation='softmax')
                a_one_hot = tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), depth=self.action_dim), axis=1)
                log_policy = tf.reduce_sum(tf.math.log(policy) * tf.stop_gradient(a_one_hot), axis=1, keepdims=True)
            else:
                dist = self.network.dist(s)
                log_policy = dist.log_prob(a)

            loss = tf.reduce_sum(-log_policy*returns)


        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        self.buffer.delete()

