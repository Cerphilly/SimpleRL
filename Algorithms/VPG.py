#Policy Gradient Methods for Reinforcement Learning with Function Approximation, Sutton et al, 2000
#High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
#https://spinningup.openai.com/en/latest/algorithms/vpg.html

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Networks.Basic_Networks import Policy_network, V_network
from Networks.Gaussian_Actor import Gaussian_Actor



class VPG:#make it useful for both discrete(cartegorical actor) and continuous actor(gaussian policy)
    def __init__(self, state_dim, action_dim, discrete=True, actor=None, critic=None, training_step=1, gamma = 0.99, lambda_gae = 0.96, learning_rate = 0.001):

        self.actor = actor
        self.critic = critic

        self.discrete = discrete

        self.buffer = Buffer()

        self.gamma = gamma
        self.lambda_gae = lambda_gae

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_start = 0
        self.training_step = training_step


        if self.actor == None:
            if self.discrete == True:
                self.actor = Policy_network(self.state_dim, self.action_dim)
            else:
                self.actor = Gaussian_Actor(self.state_dim, self.action_dim)

        if self.critic == None:
            self.critic = V_network(self.state_dim)

        self.network_list = {'Actor': self.actor, 'Critic': self.critic}
        self.name = 'VPG'

    def get_action(self, state):
        state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        if self.discrete == True:
            policy = self.actor(state, activation='softmax').numpy()[0]
            action = np.random.choice(self.action_dim, 1, p=policy)[0]
        else:
            action = self.actor(state).numpy()[0]
            action = np.clip(action, -1, 1)

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

        with tf.GradientTape(persistent=True) as tape:
            if self.discrete == True:
                policy = self.actor(s, activation='softmax')
                a_one_hot = tf.squeeze(tf.one_hot(tf.cast(a, tf.int32), depth=self.action_dim), axis=1)
                log_policy = tf.reduce_sum(tf.math.log(policy) * tf.stop_gradient(a_one_hot), axis=1, keepdims=True)
            else:
                dist = self.actor.dist(s)
                log_policy = dist.log_prob(a)

            actor_loss = -tf.reduce_sum(log_policy * tf.stop_gradient(advantages))
            critic_loss = 0.5 * tf.reduce_mean(tf.square(tf.stop_gradient(returns) - self.critic(s)))

        actor_variables = self.actor.trainable_variables
        critic_variables = self.critic.trainable_variables

        actor_gradients = tape.gradient(actor_loss, actor_variables)
        critic_gradients = tape.gradient(critic_loss, critic_variables)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, actor_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, critic_variables))

        self.buffer.delete()







