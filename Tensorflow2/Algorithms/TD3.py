#Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018.

import tensorflow as tf
import numpy as np

from Tensorflow2.Common.TFBuffer import TFBuffer
from Tensorflow2.Common.TFSaver import TFSaver
from Tensorflow2.Common.Utils import copy_weight, soft_update
from Tensorflow2.Networks.Basic_Networks import Policy_network, Q_network

class TD3:
    def __init__(self, state_dim, action_dim, max_action, min_action, save, load, actor = None, target_actor = None, critic1 = None, target_critic1=None, critic2 = None, target_critic2=None, batch_size=100, buffer_size=1e6,
                 gamma=0.99, tau=0.005, actor_lr=0.001, critic_lr=0.001, policy_delay=2, actor_noise=0.1, target_noise=0.2, noise_clip=0.5, training_start=500):

        self.actor = actor
        self.target_actor = target_actor
        self.critic1 = critic1
        self.target_critic1 = target_critic1
        self.critic2 = critic2
        self.target_critic2 = target_critic2

        self.buffer = TFBuffer(buffer_size)
        self.saver = TFSaver('TD3', 'test')

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.save = save
        self.load = load

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.policy_delay = policy_delay
        self.actor_noise = actor_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.training_start = training_start

        self.step = 0

        if self.actor == None:
            self.actor = Policy_network(self.state_dim, self.action_dim)
        if self.target_actor == None:
            self.target_actor = Policy_network(self.state_dim, self.action_dim)
        if self.critic1 == None:
            self.critic1 = Q_network(self.state_dim, self.action_dim)
        if self.target_critic1 == None:
            self.target_critic1 = Q_network(self.state_dim, self.action_dim)
        if self.critic2 == None:
            self.critic2 = Q_network(self.state_dim, self.action_dim)
        if self.target_critic2 == None:
            self.target_critic2 = Q_network(self.state_dim, self.action_dim)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2}


    def get_action(self, state):
        state = np.array(state)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        noise = np.random.normal(loc=0, scale=self.actor_noise, size=self.action_dim)

        action = self.actor(state).numpy()[0] + noise

        action = self.max_action * np.clip(action, self.min_action, self.max_action)

        return action

    def train(self, training_num):
        self.actor_loss = 0
        self.critic1_loss = 0
        self.critic2_loss = 0

        for i in range(training_num):
            self.step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)

            target_action = tf.clip_by_value(self.target_actor(ns) + tf.clip_by_value(tf.random.normal(shape=self.target_actor(ns).shape, mean=0, stddev=self.target_noise), -self.noise_clip, self.noise_clip), self.min_action, self.max_action)

            target_value = tf.stop_gradient(r + self.gamma * (1 - d) * tf.minimum(self.target_critic1(ns, target_action), self.target_critic2(ns, target_action)))

            with tf.GradientTape(persistent=True) as tape:
                critic1_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic1(s, a)))
                critic2_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic2(s, a)))

            critic1_grad = tape.gradient(critic1_loss, self.critic1.trainable_variables)
            self.critic1_optimizer.apply_gradients(zip(critic1_grad, self.critic1.trainable_variables))

            critic2_grad = tape.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_grad, self.critic2.trainable_variables))

            if self.step % self.policy_delay == 0:

                with tf.GradientTape() as tape2:
                    actor_loss = -tf.reduce_mean(self.critic1(s, self.actor(s)))

                actor_grad = tape2.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

                soft_update(self.actor, self.target_actor, self.tau)

                self.actor_loss += actor_loss.numpy()

                del tape2

            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)

            self.critic1_loss += critic1_loss.numpy()
            self.critic2_loss += critic2_loss.numpy()

            del tape


