#Proximal Policy Optimization Algorithms, Schulman et al, 2017
#Emergence of Locomotion Behaviours in Rich Environments, Heess et al

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from Common.Buffer import Buffer
from Networks.Basic_Networks import Policy_network, V_network


class PPO:#make it useful for both discrete(cartegorical actor) and continuous actor(gaussian policy)
    def __init__(self, state_dim, action_dim, max_action = 1, min_action=1, discrete=True, actor=None, critic=None, gamma = 0.99, lambda_gae = 0.95, learning_rate = 3e-4, batch_size=64, num_epoch=10):
        self.actor = actor
        self.critic = critic
        self.max_action = max_action
        self.min_action = min_action

        self.discrete = discrete

        self.buffer = Buffer()

        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.batch_size = batch_size
        self.num_epoch = num_epoch

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0

        if self.actor == None:
            if self.discrete == True:
                self.actor = Policy_network(self.state_dim, self.action_dim)
            else:
                self.actor = Policy_network(self.state_dim, self.action_dim*2)

        if self.critic == None:
            self.critic = V_network(self.state_dim)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}

    def get_action(self, state):
        state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        if self.discrete == True:
            policy = self.actor(state, activation='softmax').numpy()[0]
            action = np.random.choice(self.action_dim, 1, p=policy)[0]
        else:
            output = self.actor(state)
            mean, log_std = self.max_action*(output[:, :self.action_dim]), output[:, self.action_dim:]
            std = tf.exp(log_std)

            eps = tf.random.normal(tf.shape(mean))
            action = (mean + std * eps)[0]
            action = tf.clip_by_value(action, self.min_action, self.max_action)


        return action

    def train(self, training_num):
        s, a, r, ns, d = self.buffer.all_sample()

        values = self.critic(s)

        returns = np.zeros_like(r.numpy())
        advantages = np.zeros_like(returns)

        running_return = np.zeros(1)
        previous_value = np.zeros(1)
        running_advantage = np.zeros(1)

        for t in reversed(range(len(r))):
            running_return = (r[t] + self.gamma * running_return * (1 - d[t])).numpy()
            running_tderror = (r[t] + self.gamma * previous_value * (1 - d[t]) - values[t]).numpy()
            running_advantage = (running_tderror + (self.gamma * self.lambda_gae) * running_advantage * (1 - d[t])).numpy()

            returns[t] = running_return
            previous_value = values[t]
            advantages[t] = running_advantage

        old_policy = self.actor(s, activation = 'softmax').numpy()
        n = len(s)
        arr = np.arange(n)

        for epoch in range(self.num_epoch):
            np.random.shuffle(arr)
            for i in range(n // self.batch_size):
                batch_index = arr[self.batch_size*i: self.batch_size*(i+1)]
                if self.discrete == True:
                    policy = self.actor(s, activation='softmax')


                    pass
                else:
                    pass
