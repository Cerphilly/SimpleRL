#Continuous Control With Deep Reinforcement Learning, Lillicrap et al, 2015.

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Networks.Basic_Networks import Policy_network, Q_network


class DDPG:
    def __init__(self, state_dim, action_dim, actor = None, target_actor = None, critic = None, target_critic=None, training_step=100, batch_size=100, buffer_size=1e6, gamma=0.99, tau = 0.005, actor_lr=0.001, critic_lr=0.001, training_start=500):
        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic

        self.buffer = Buffer(buffer_size)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim


        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.training_start = training_start
        self.training_step = training_step


        if self.actor == None:
            self.actor = Policy_network(self.state_dim, self.action_dim)
        if self.target_actor == None:
            self.target_actor = Policy_network(self.state_dim, self.action_dim)
        if self.critic == None:
            self.critic = Q_network(self.state_dim, self.action_dim)
        if self.target_critic == None:
            self.target_critic = Q_network(self.state_dim, self.action_dim)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic, self.target_critic)

        self.network_list = {'Actor': self.actor, 'Target_Actor': self.target_actor, 'Critic': self.critic, 'Target_Critic': self.target_critic}
        self.name = 'DDPG'

    def get_action(self, state):
        state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        noise = np.random.normal(loc=0, scale=0.1, size = self.action_dim)
        action = self.actor(state).numpy()[0] + noise

        action = np.clip(action, -1, 1)

        return action

    def train(self, training_num):
        self.actor_loss = 0
        self.critic_loss = 0

        for i in range(training_num):

            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            value_next = tf.stop_gradient(self.target_critic(ns, self.target_actor(ns)))
            target_value = r + (1 - d) * self.gamma * value_next

            with tf.GradientTape(persistent=True) as tape:
                critic_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic(s, a)))
                actor_loss = -tf.reduce_mean(self.critic(s, self.actor(s)))

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients((zip(actor_grad, self.actor.trainable_variables)))

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic, self.target_critic, self.tau)














