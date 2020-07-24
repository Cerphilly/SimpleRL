#Simple statistical gradient-following algorithms for connectionist reinforcement learning, Ronald J. Williams, 1992

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym
from Common.Buffer import Buffer
from Networks.Basic_Networks import Policy_network
from Networks.Gaussian_Actor import Gaussian_Actor


class REINFORCE:
    def __init__(self, state_dim, action_dim, max_action = 1, min_action=1, discrete=True, network=None, gamma=0.99, learning_rate=0.001):
        self.network = network

        self.buffer = Buffer()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        
        self.discrete = discrete

        self.gamma = gamma
        self.training_start = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        if network == None:
            if discrete == True:
                self.network = Policy_network(self.state_dim, self.action_dim)
            else:
                self.network = Policy_network(self.state_dim, self.action_dim*2)

        self.network_list = {'Network': self.network}


    def get_action(self, state):
        state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
            
        if self.discrete == True:
            policy = self.network(state, activation='softmax').numpy()[0]
            action = np.random.choice(self.action_dim, 1, p=policy)[0]
            
        else:
            # mean = self.network(state, activation='linear').numpy()[0]
            # logstd = tf.zeros_like(mean)
            # std = tf.exp(logstd)
            # action = tf.clip_by_value(tf.random.normal(tf.shape(mean), mean, std), self.min_action, self.max_action)

            output = self.network(state, activation='linear')
            mean, log_std = output[:, :self.action_dim], output[:, self.action_dim:]
            std = tf.exp(log_std)

            eps = tf.random.normal(tf.shape(mean))

            action = (mean + std*eps)[0]
            action = tf.clip_by_value(action, self.min_action, self.max_action)

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
                output = self.network(s, activation='linear')
                mean, log_std = output[:, :self.action_dim], output[:, self.action_dim:]
                std = tf.exp(log_std)
                dist = tfp.distributions.Normal(loc=mean, scale=std)
                log_policy = dist.log_prob(a)

            loss = tf.reduce_sum(-log_policy*returns)


        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        self.buffer.delete()

