#Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018.

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Network.Basic_Networks import Q_network, V_network
from Network.Gaussian_Actor import Squashed_Gaussian_Actor

class SAC_v1:
    def __init__(self, state_dim, action_dim, args):

        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim, max_size=args.buffer_size, on_policy=False)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.v_network_optimizer = tf.keras.optimizers.Adam(args.v_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0

        self.actor = Squashed_Gaussian_Actor(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim,
                                             log_std_min=args.log_std_min, log_std_max=args.log_std_max, activation=args.activation)
        self.critic1 = Q_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.critic2 = Q_network(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.v_network = V_network(state_dim=self.state_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_v_network = V_network(state_dim=self.state_dim,
                                          hidden_units=args.hidden_dim, activation=args.activation)

        copy_weight(self.v_network, self.target_v_network)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'V_network': self.v_network, 'Target_V_network': self.target_v_network}
        self.name = 'SAC_v1'

    def get_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        action, _ = self.actor(state)
        action = np.clip(action.numpy()[0], -1, 1)

        return action

    def eval_action(self, state):
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        action, _ = self.actor(state, deterministic=True)
        action = np.clip(action.numpy()[0], -1, 1)

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_v_loss = 0
        for i in range(training_num):
            self.current_step += 1

            #v_network training
            s, a, r, ns, d = self.buffer.sample(self.batch_size)
            s_action, s_logpi = self.actor(s)
            min_aq = tf.minimum(self.critic1(s, s_action), self.critic2(s, s_action))
            target_v = tf.stop_gradient(min_aq - self.alpha * s_logpi)

            with tf.GradientTape() as tape1:
                v_loss = 0.5 * tf.reduce_mean(tf.square(self.v_network(s) - target_v))

            v_gradients = tape1.gradient(v_loss, self.v_network.trainable_variables)
            self.v_network_optimizer.apply_gradients(zip(v_gradients, self.v_network.trainable_variables))

            #critic network training
            target_q = tf.stop_gradient(r + self.gamma * (1 - d) * self.target_v_network(ns))

            with tf.GradientTape(persistent=True) as tape2:
                critic1_loss = 0.5 * tf.reduce_mean(tf.square(self.critic1(s, a) - target_q))
                critic2_loss = 0.5 * tf.reduce_mean(tf.square(self.critic2(s, a) - target_q))

            critic1_gradients = tape2.gradient(critic1_loss, self.critic1.trainable_variables)
            self.critic1_optimizer.apply_gradients(zip(critic1_gradients, self.critic1.trainable_variables))

            critic2_gradients = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_gradients, self.critic2.trainable_variables))

            #actor network training
            with tf.GradientTape() as tape3:
                s_action, s_logpi = self.actor(s)

                min_aq_rep = tf.minimum(self.critic1(s, s_action), self.critic2(s, s_action))
                actor_loss = tf.reduce_mean(self.alpha * s_logpi - min_aq_rep)

            actor_grad = tape3.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            #v network update
            soft_update(self.v_network, self.target_v_network, self.tau)

            del tape1, tape2, tape3

            total_a_loss += actor_loss.numpy()
            total_c1_loss += critic1_loss.numpy()
            total_c2_loss += critic2_loss.numpy()
            total_v_loss += v_loss.numpy()


        return {'Loss': {'Actor': total_a_loss, 'Critic1': total_c1_loss, 'Critic2': total_c2_loss, 'V': total_v_loss}}



