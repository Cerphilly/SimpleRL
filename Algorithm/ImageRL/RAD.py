#RAD: Reinforcement Learning with Augmented Data, Laskin et al, 2020
#Reference: https://github.com/MishaLaskin/rad (official repo)

import tensorflow as tf
import numpy as np

from Network.Gaussian_Actor import Squashed_Gaussian_Actor
from Network.Basic_Networks import Q_network
from Network.Encoder import PixelEncoder

from Common.Utils import copy_weight, soft_update
from Common.Buffer import Buffer
from Common.Data_Augmentation import *

class RADBuffer(Buffer):
    def rad_sample(self, batch_size, aug_funcs, image_size=84):
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)

        states = self.s[ids]
        actions = self.a[ids]
        rewards = self.r[ids]
        states_next = self.ns[ids]
        dones = self.d[ids]

        for aug, func in aug_funcs.items():
            if 'crop' in aug:
                states = func(states, output_size=image_size)
                states_next = func(states_next, output_size=image_size)

            elif 'translate' in aug:
                # states = center_crop(states, image_size)
                # states_next = center_crop(states_next, image_size)

                states, random_idxs = func(states, output_size=image_size, return_random_idxs=True)
                states_next = func(states_next, output_size=image_size, h1s=random_idxs['h1s'], w1s=random_idxs['w1s'])

            elif 'cutout' in aug:
                states = func(states)
                states_next = func(states_next)

        for aug, func in aug_funcs.items():
            if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                continue
            states = func(states)
            states_next = func(states_next)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_next = tf.convert_to_tensor(states_next, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        if self.on_policy:
            log_probs = self.log_prob[ids]
            log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)

            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones

class RAD_SACv2:
    def __init__(self, obs_dim, action_dim, args):

        self.data_augs = args.data_augs

        self.buffer = RADBuffer(state_dim=(obs_dim[0], args.pre_image_size, args.pre_image_size), action_dim=action_dim, max_size=args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = args.image_size
        self.pre_image_size = args.pre_image_size
        self.current_step = 0

        self.log_alpha = tf.Variable(initial_value=tf.math.log(args.alpha), trainable=args.train_alpha)
        self.target_entropy = -action_dim
        self.gamma = args.gamma

        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim

        self.tau = args.tau
        self.encoder_tau = args.encoder_tau
        self.critic_update = args.critic_update

        self.training_start = args.training_start
        self.training_step = args.training_step
        self.train_alpha = args.train_alpha

        self.actor = Squashed_Gaussian_Actor(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, log_std_min=args.log_std_min, log_std_max=args.log_std_max,
                                             activation=args.activation)
        self.critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim,
                                activation=args.activation)
        self.critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim,
                                activation=args.activation)
        self.target_critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim,
                                      activation=args.activation)
        self.target_critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim,
                                      activation=args.activation)

        self.encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, activation=args.activation)
        self.target_encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, activation=args.activation)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.log_alpha_optimizer = tf.keras.optimizers.Adam(args.alpha_lr, beta_1=0.5)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder}

        self.aug_funcs = {}
        self.aug_list = {
            'crop': random_crop,
            'translate': random_translate,
            'grayscale': grayscale,
            'cutout': cutout,
            'cutout_color': cutout_color,
            'flip': flip,
            'rotate': rotation,
            'rand_conv': convolution,
            'color_jitter': random_color_jitter(batch_size=self.batch_size, stack_size=args.frame_stack),
            'no_aug': no_aug
        }
        for aug_name in args.data_augs.split('-'):
            assert aug_name in self.aug_list
            self.aug_funcs[aug_name] = self.aug_list[aug_name]

        self.name = 'RAD_SACv2'

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)


    def get_action(self, obs):
        if obs.shape[-1] != self.image_size:
            if 'crop' in self.data_augs:
                obs = center_crop(obs, self.image_size)
            elif 'translate' in self.data_augs:
                obs = center_translate(obs, self.image_size)

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature)
        action = action.numpy()[0]

        return action

    def eval_action(self, obs):
        if obs.shape[-1] != self.image_size:
            if 'crop' in self.data_augs:
                obs = center_crop(obs, self.image_size)
            elif 'translate' in self.data_augs:
                obs = center_translate(obs, self.image_size)

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature, deterministic=True)
        action = action.numpy()[0]

        return action

    def train(self, training_num):

        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_alpha_loss = 0

        for i in range(training_num):
            self.current_step += 1

            s, a, r, ns, d = self.buffer.rad_sample(self.batch_size, self.aug_funcs, self.image_size)

            #critic network training
            ns_action, ns_logpi = self.actor(self.encoder(ns))

            target_min_aq = tf.minimum(self.target_critic1(self.target_encoder(ns), ns_action),
                                       self.target_critic2(self.target_encoder(ns), ns_action))

            target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (
                    target_min_aq - self.alpha.numpy() * ns_logpi))

            with tf.GradientTape(persistent=True) as tape1:
                critic1_loss = tf.reduce_mean(tf.square(self.critic1(self.encoder(s), a) - target_q))
                critic2_loss = tf.reduce_mean(tf.square(self.critic2(self.encoder(s), a) - target_q))

            critic1_gradients = tape1.gradient(critic1_loss,
                                               self.encoder.trainable_variables + self.critic1.trainable_variables)

            critic2_gradients = tape1.gradient(critic2_loss,
                                               self.encoder.trainable_variables + self.critic2.trainable_variables)

            self.critic1_optimizer.apply_gradients(
                zip(critic1_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables))

            self.critic2_optimizer.apply_gradients(
                zip(critic2_gradients, self.encoder.trainable_variables + self.critic2.trainable_variables))

            total_c1_loss += critic1_loss.numpy()
            total_c2_loss += critic2_loss.numpy()

            del tape1

            #actor network training
            with tf.GradientTape() as tape2:
                s_action, s_logpi = self.actor(tf.stop_gradient(self.encoder(s)))

                min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), s_action),
                                        self.critic2(tf.stop_gradient(self.encoder(s)), s_action))

                actor_loss = tf.reduce_mean(self.alpha.numpy() * s_logpi - min_aq_rep)

            actor_gradients = tape2.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            total_a_loss += actor_loss.numpy()

            del tape2

            #alpha training
            if self.train_alpha:
                with tf.GradientTape() as tape3:
                    _, s_logpi = self.actor(self.encoder(s))
                    alpha_loss = -tf.exp(self.log_alpha) * tf.stop_gradient(s_logpi + self.target_entropy)
                    alpha_loss = tf.nn.compute_average_loss(alpha_loss)
                    #alpha_loss = tf.reduce_mean(alpha_loss)

                log_alpha_gradients = tape3.gradient(alpha_loss, [self.log_alpha])
                self.log_alpha_optimizer.apply_gradients(zip(log_alpha_gradients, [self.log_alpha]))

                total_alpha_loss += alpha_loss.numpy()

                del tape3

            #target network update
            if self.current_step % self.critic_update == 0:
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)
                soft_update(self.encoder, self.target_encoder, self.encoder_tau)

        return {'Loss': {'Actor': total_a_loss, 'Critic1': total_c1_loss, 'Critic2': total_c2_loss, 'Alpha': total_alpha_loss},
                'Value': {'Alpha': tf.exp(self.log_alpha).numpy()}}
