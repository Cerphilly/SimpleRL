#Soft Actor-Critic Algorithms and Applications, Haarnoja et al, 2018

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Networks.Basic_Networks import Q_network
from Networks.Gaussian_Actor import Squashed_Gaussian_Actor


class SAC_v2:
    def __init__(self, state_dim, action_dim, actor=None, critic1=None, target_critic1=None, critic2=None, target_critic2=None, training_step=200,
                 batch_size=100, buffer_size=1e6, tau=0.005, learning_rate=0.0003, gamma=0.99, alpha=0.2, auto_alpha = False, reward_scale=1, training_start = 500):

        self.actor = actor
        self.critic1 = critic1
        self.target_critic1 = target_critic1
        self.critic2 = critic2
        self.target_critic2 = target_critic2

        self.buffer = Buffer(buffer_size)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.auto_alpha = auto_alpha
        self.reward_scale = reward_scale
        self.training_start = training_start
        self.training_step = training_step

        if auto_alpha == False:
            self.alpha = alpha
        else:
            #self.log_alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha = tf.Variable(0.2, dtype=tf.float32)
            self.target_alpha = -action_dim
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate)

        if self.actor == None:
            self.actor = Squashed_Gaussian_Actor(self.state_dim, self.action_dim)
        if self.critic1 == None:
            self.critic1 = Q_network(self.state_dim, self.action_dim)
        if self.target_critic1 == None:
            self.target_critic1 = Q_network(self.state_dim, self.action_dim)
        if self.critic2 == None:
            self.critic2 = Q_network(self.state_dim, self.action_dim)
        if self.target_critic2 == None:
            self.target_critic2 = Q_network(self.state_dim, self.action_dim)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}
        self.name = 'SAC_v2'

    def get_action(self, state):
        state = np.expand_dims(np.array(state), axis=0)

        action = self.actor(state).numpy()[0]

        return action

    def train(self, training_num):


        for i in range(training_num):
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            with tf.GradientTape(persistent=True) as tape:
                target_min_aq = tf.minimum(self.target_critic1(ns, self.actor(ns)), self.target_critic2(ns, self.actor(ns)))

                target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (target_min_aq - self.alpha * self.actor.log_pi(ns)))

                critic1_loss = tf.reduce_mean(tf.square(self.critic1(s, a) - target_q))
                critic2_loss = tf.reduce_mean(tf.square(self.critic2(s, a) - target_q))

                min_aq = tf.minimum(self.critic1(s, self.actor(s)), self.critic2(s, self.actor(s)))

                mu, sigma = self.actor.mu_sigma(s)
                output = mu + tf.random.normal(shape=mu.shape) * sigma

                min_aq_rep = self.critic1(s, output)

                actor_loss = tf.reduce_mean(self.alpha * self.actor.log_pi(s) - min_aq_rep)

                if self.auto_alpha == True:
                    alpha_loss = -tf.reduce_mean(self.alpha*(tf.stop_gradient(self.actor.log_pi(s) + self.target_alpha)))


            critic1_gradients = tape.gradient(critic1_loss, self.critic1.trainable_variables)
            self.critic1_optimizer.apply_gradients(zip(critic1_gradients, self.critic1.trainable_variables))
            critic2_gradients = tape.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_gradients, self.critic2.trainable_variables))

            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            if self.auto_alpha == True:
                alpha_grad = tape.gradient(alpha_loss, [self.alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.alpha]))
                #self.alpha.assign(tf.exp(self.log_alpha))

            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)



