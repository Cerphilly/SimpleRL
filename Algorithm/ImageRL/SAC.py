#Soft Actor-Critic Algorithm and Applications, Haarnoja et al, 2018

import tensorflow as tf
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update, find_channel
from Network.Basic_Networks import Q_network
from Network.Gaussian_Actor import Squashed_Gaussian_Actor
from Network.Encoder import PixelEncoder

class ImageSAC_v2:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = Buffer(state_dim=obs_dim, action_dim=action_dim, max_size=args.buffer_size)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        # self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        # self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.current_step = 0
        self.critic_update = args.critic_update

        self.feature_dim = args.feature_dim

        self.tau = args.tau
        self.encoder_tau = args.encoder_tau

        self.log_alpha = tf.Variable(np.log(args.alpha), dtype=tf.float32, trainable=args.train_alpha)
        self.target_entropy = -action_dim
        self.alpha_optimizer = tf.keras.optimizers.Adam(args.alpha_lr)
        self.train_alpha = args.train_alpha

        self.actor = Squashed_Gaussian_Actor(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim,
                                             log_std_min=args.log_std_min, log_std_max=args.log_std_max, activation=args.activation)
        self.critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)

        self.encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, data_format=find_channel(args.domain_type), activation=args.activation)
        self.target_encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, data_format=find_channel(args.domain_type), activation=args.activation)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)


        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder}
        self.name = 'ImageSAC_v2'

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, observation):
        observation = np.expand_dims(np.array(observation), axis=0)
        feature = self.encoder(observation)
        action, _ = self.actor(feature)
        action = np.clip(action.numpy()[0], -1, 1)

        return action

    def eval_action(self, observation):
        observation = np.expand_dims(np.array(observation), axis=0)
        feature = self.encoder(observation)
        action, _ = self.actor(feature, deterministic=True)
        action = np.clip(action.numpy()[0], -1, 1)

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_alpha_loss = 0

        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.sample(self.batch_size)
            #critic network training
            ns_action, ns_logpi = self.actor(self.encoder(ns))

            target_min_aq = tf.minimum(self.target_critic1(self.target_encoder(ns), ns_action), self.target_critic2(self.target_encoder(ns), ns_action))

            target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (target_min_aq - self.alpha.numpy() * ns_logpi))
            with tf.GradientTape() as tape1:
                critic1_loss = tf.reduce_mean(tf.square(self.critic1(self.encoder(s), a) - target_q))
                critic2_loss = tf.reduce_mean(tf.square(self.critic2(self.encoder(s), a) - target_q))
                critic_loss = critic1_loss + critic2_loss

            # critic1_gradients = tape1.gradient(critic1_loss, self.encoder.trainable_variables + self.critic1.trainable_variables)
            # critic2_gradients = tape1.gradient(critic2_loss, self.encoder.trainable_variables + self.critic2.trainable_variables)
            critic_gradients = tape1.gradient(critic_loss, self.encoder.trainable_variables +
                                              self.critic1.trainable_variables + self.critic2.trainable_variables)

            # self.critic1_optimizer.apply_gradients(zip(critic1_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables))
            # self.critic2_optimizer.apply_gradients(zip(critic2_gradients, self.encoder.trainable_variables + self.critic2.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.encoder.trainable_variables +
                                                      self.critic1.trainable_variables + self.critic2.trainable_variables))
            total_c1_loss += critic1_loss.numpy()
            total_c2_loss += critic2_loss.numpy()

            del tape1

            #actor network training
            with tf.GradientTape() as tape2:
                s_action, s_logpi = self.actor(tf.stop_gradient(self.encoder(s)))

                min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), s_action), self.critic2(tf.stop_gradient(self.encoder(s)), s_action))

                actor_loss = tf.reduce_mean(self.alpha.numpy() * s_logpi - min_aq_rep)

            actor_gradients = tape2.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            total_a_loss += actor_loss.numpy()

            del tape2

            #alpha training
            if self.train_alpha:
                with tf.GradientTape() as tape3:
                    _, s_logpi = self.actor(tf.stop_gradient(self.encoder(s)))
                    alpha_loss = -(tf.exp(self.log_alpha) * (tf.stop_gradient(s_logpi + self.target_entropy)))
                    alpha_loss = tf.nn.compute_average_loss(alpha_loss)#from softlearning package

                alpha_grad = tape3.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))

                total_alpha_loss += alpha_loss.numpy()

                del tape3

            #target network update
            if self.current_step % self.critic_update == 0:
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)
                soft_update(self.encoder, self.target_encoder, self.encoder_tau)


        return {'Loss': {'Actor': total_a_loss, 'Critic1': total_c1_loss, 'Critic2': total_c2_loss, 'Alpha': total_alpha_loss},
                'Value': {'Alpha': tf.exp(self.log_alpha).numpy()}}



