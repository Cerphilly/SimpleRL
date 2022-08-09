#CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning, Srinivas et al, 2020

import tensorflow as tf
import numpy as np

from Network.Gaussian_Actor import Squashed_Gaussian_Actor
from Network.Basic_Networks import Q_network, V_network, Policy_network
from Network.Encoder import PixelEncoder
from Network.CURL import CURL

from Common.Utils import copy_weight, soft_update
from Common.Buffer import Buffer
from Common.Data_Augmentation import center_crop, random_crop

class CURLBuffer(Buffer):
    def cpc_sample(self, batch_size, image_size=84, data_format='channels_first'):
        # ImageRL/CURL
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)

        states = self.s[ids]
        actions = self.a[ids]
        rewards = self.r[ids]
        states_next = self.ns[ids]
        dones = self.d[ids]

        pos = states.copy()

        states = random_crop(states, image_size, data_format)
        states_next = random_crop(states_next, image_size, data_format)
        pos = random_crop(pos, image_size, data_format)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_next = tf.convert_to_tensor(states_next, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        cpc_kwargs = dict(obs_anchor=states, obs_pos=pos, time_anchor=None, time_pos=None)

        if self.on_policy == True:
            log_probs = self.log_prob[ids]
            log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)

            return states, actions, rewards, states_next, dones, log_probs, cpc_kwargs

        return states, actions, rewards, states_next, dones, cpc_kwargs


class CURL_SACv1:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = CURLBuffer(state_dim=(obs_dim[0], args.pre_image_size, args.pre_image_size), action_dim=action_dim, max_size=args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]

        self.gamma = args.gamma
        self.alpha = args.alpha

        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim
        self.curl_latent_dim = args.curl_latent_dim

        self.tau = args.tau
        self.encoder_tau = args.encoder_tau

        self.training_start = args.training_start
        self.training_step = args.training_step

        self.actor = Squashed_Gaussian_Actor(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim,
                                             log_std_min=args.log_std_min, log_std_max=args.log_std_max, activation=args.activation)
        self.critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.v_network = V_network(state_dim=self.feature_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_v_network = V_network(state_dim=self.feature_dim, hidden_units=args.hidden_dim, activation=args.activation)

        self.encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, activation=args.activation)
        self.target_encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, activation=args.activation)

        self.curl = CURL(z_dim=self.feature_dim, batch_size=self.curl_latent_dim)

        copy_weight(self.v_network, self.target_v_network)
        copy_weight(self.encoder, self.target_encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        #self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.v_network_optimizer = tf.keras.optimizers.Adam(args.v_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.cpc_optimizer = tf.keras.optimizers.Adam(args.cpc_lr)

        self.current_step = 0

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'V_network': self.v_network,
                             'Target_V_network': self.target_v_network, 'Curl':self.curl, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder}

        self.name = 'CURL_SACv1'

    def get_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature)
        action = action.numpy()[0]

        return action

    def eval_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature, deterministic=True)
        action = action.numpy()[0]

        return action

    def train(self, training_step):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_v_loss = 0
        total_cpc_loss = 0

        for i in range(training_step):
            self.current_step += 1

            s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size)

            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            #v network training
            s_action, s_logpi = self.actor(self.encoder(s))

            min_aq = tf.minimum(self.critic1(self.encoder(s), s_action), self.critic2(self.encoder(s), s_action))
            target_v = tf.stop_gradient(min_aq - self.alpha * s_logpi)

            with tf.GradientTape() as tape1:
                v_loss = tf.reduce_mean(tf.square(self.v_network(tf.stop_gradient(self.encoder(s))) - target_v))

            v_gradients = tape1.gradient(v_loss, self.v_network.trainable_variables)
            self.v_network_optimizer.apply_gradients(zip(v_gradients, self.v_network.trainable_variables))

            total_v_loss += v_loss.numpy()

            del tape1
            #critic network training
            target_q = tf.stop_gradient(r + self.gamma * (1 - d) * self.target_v_network(self.target_encoder(ns)))

            with tf.GradientTape(persistent=True) as tape2:
                critic1_loss = tf.reduce_mean(tf.square(self.critic1(self.encoder(s), a) - target_q))
                critic2_loss = tf.reduce_mean(tf.square(self.critic2(self.encoder(s), a) - target_q))

                #critic_loss = critic1_loss + critic2_loss

            critic1_gradients = tape2.gradient(critic1_loss, self.encoder.trainable_variables + self.critic1.trainable_variables)
            critic2_gradients = tape2.gradient(critic2_loss, self.encoder.trainable_variables + self.critic2.trainable_variables)

            # critic_gradients = tape2.gradient(critic_loss, self.encoder.trainable_variables + self.critic1.trainable_variables
            #                                   + self.critic2.trainable_variables)

            self.critic1_optimizer.apply_gradients(zip(critic1_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables))
            self.critic2_optimizer.apply_gradients(zip(critic2_gradients, self.encoder.trainable_variables + self.critic2.trainable_variables))

            # self.critic_optimizer.apply_gradients(zip(critic_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables
            #                                           + self.critic2.trainable_variables))

            total_c1_loss += critic1_loss.numpy()
            total_c2_loss += critic2_loss.numpy()

            del tape2
            #actor network training
            with tf.GradientTape() as tape3:
                s_action, s_logpi = self.actor(tf.stop_gradient(self.encoder(s)))

                min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), s_action),
                                        self.critic2(tf.stop_gradient(self.encoder(s)), s_action))

                actor_loss = tf.reduce_mean(self.alpha * s_logpi - min_aq_rep)

            actor_gradients = tape3.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            total_a_loss += actor_loss.numpy()

            #curl and encoder network training
            with tf.GradientTape(persistent=True) as tape4:
                z_a = self.encoder(obs_anchor)
                z_pos = tf.stop_gradient(self.target_encoder(obs_pos))

                logits = self.curl.compute_logits(z_a, z_pos)
                labels = tf.range(logits.shape[0], dtype='int64')

                cpc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

            cpc_gradients = tape4.gradient(cpc_loss, self.curl.trainable_variables)
            self.cpc_optimizer.apply_gradients((zip(cpc_gradients, self.curl.trainable_variables)))

            encoder_gradients = tape4.gradient(cpc_loss, self.encoder.trainable_variables)
            self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))

            total_cpc_loss += cpc_loss.numpy()

            del tape4

            soft_update(self.v_network, self.target_v_network, self.tau)
            soft_update(self.encoder, self.target_encoder, self.encoder_tau)

        return {'Loss': {'Actor': total_a_loss, 'Critic1': total_c1_loss, 'Critic2': total_c2_loss, 'V': total_v_loss,
                         'CPC': total_cpc_loss}}




class CURL_SACv2:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = CURLBuffer(state_dim=(obs_dim[0], args.pre_image_size, args.pre_image_size), action_dim=action_dim, max_size=args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]
        self.current_step = 0

        self.log_alpha = tf.Variable(initial_value=tf.math.log(args.alpha), trainable=args.train_alpha)
        self.target_entropy = -action_dim
        self.gamma = args.gamma

        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim
        self.curl_latent_dim = args.curl_latent_dim

        self.tau = args.tau
        self.encoder_tau = args.encoder_tau
        self.critic_update = args.critic_update

        self.training_start = args.training_start
        self.training_step = args.training_step
        self.train_alpha = args.train_alpha

        self.actor = Squashed_Gaussian_Actor(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim,
                                             log_std_min=args.log_std_min, log_std_max=args.log_std_max, activation=args.activation)
        self.critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)

        self.encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, activation=args.activation)
        self.target_encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, activation=args.activation)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.curl = CURL(z_dim=self.feature_dim, batch_size=self.curl_latent_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        #self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.cpc_optimizer = tf.keras.optimizers.Adam(args.cpc_lr)
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(args.alpha_lr, beta_1=0.5)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2, 'Curl':self.curl, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder}

        self.name = 'CURL_SACv2'

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature)
        action = action.numpy()[0]

        return action

    def eval_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature, deterministic=True)
        action = action.numpy()[0]

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_cpc_loss = 0
        total_alpha_loss = 0

        for i in range(training_num):
            self.current_step += 1

            s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size)

            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            #critic network training
            ns_action, ns_logpi = self.actor(self.encoder(ns))

            target_min_aq = tf.minimum(self.target_critic1(self.target_encoder(ns), ns_action),
                                       self.target_critic2(self.target_encoder(ns), ns_action))

            target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (
                    target_min_aq - self.alpha.numpy() * ns_logpi))

            with tf.GradientTape(persistent=True) as tape1:
                critic1_loss = tf.reduce_mean(tf.square(self.critic1(self.encoder(s), a) - target_q))
                critic2_loss = tf.reduce_mean(tf.square(self.critic2(self.encoder(s), a) - target_q))

                #critic_loss = critic1_loss + critic2_loss

            critic1_gradients = tape1.gradient(critic1_loss,
                                               self.encoder.trainable_variables + self.critic1.trainable_variables)

            critic2_gradients = tape1.gradient(critic2_loss,
                                               self.encoder.trainable_variables + self.critic2.trainable_variables)

            self.critic1_optimizer.apply_gradients(
                zip(critic1_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables))

            self.critic2_optimizer.apply_gradients(
                zip(critic2_gradients, self.encoder.trainable_variables + self.critic2.trainable_variables))

            # critic_gradients = tape1.gradient(critic_loss, self.encoder.trainable_variables + self.critic1.trainable_variables
            #                                   + self.critic2.trainable_variables)
            #
            # self.critic_optimizer.apply_gradients(zip(critic_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables
            #                                   + self.critic2.trainable_variables))

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
                    alpha_loss = -(tf.exp(self.log_alpha) * tf.stop_gradient(s_logpi + self.target_entropy))
                    alpha_loss = tf.nn.compute_average_loss(alpha_loss)
                    #alpha_loss = tf.reduce_mean(alpha_loss)

                log_alpha_gradients = tape3.gradient(alpha_loss, [self.log_alpha])
                self.log_alpha_optimizer.apply_gradients(zip(log_alpha_gradients, [self.log_alpha]))

                total_alpha_loss += alpha_loss.numpy()

                del tape3

            #curl and encoder training
            with tf.GradientTape(persistent=True) as tape4:
                z_a = self.encoder(obs_anchor)
                z_pos = tf.stop_gradient(self.target_encoder(obs_pos))
                logits = self.curl.compute_logits(z_a, z_pos)
                labels = tf.range(logits.shape[0], dtype='int64')

                cpc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

            cpc_gradients = tape4.gradient(cpc_loss, self.curl.trainable_variables)
            self.cpc_optimizer.apply_gradients(zip(cpc_gradients, self.curl.trainable_variables))

            encoder_gradients = tape4.gradient(cpc_loss, self.encoder.trainable_variables)
            self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))

            total_cpc_loss += cpc_loss.numpy()

            del tape4

            if self.current_step % self.critic_update == 0:
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)
                soft_update(self.encoder, self.target_encoder, self.encoder_tau)

        return {'Loss': {'Actor': total_a_loss, 'Critic1': total_c1_loss, 'Critic2': total_c2_loss, 'CPC': total_cpc_loss, 'Alpha': total_alpha_loss},
                'Value': {'Alpha': tf.exp(self.log_alpha).numpy()}}


class CURL_TD3:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = CURLBuffer(state_dim=(obs_dim[0], args.pre_image_size, args.pre_image_size), action_dim=action_dim, max_size=args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]

        self.current_step = 0

        self.gamma = args.gamma

        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim
        self.curl_latent_dim = args.curl_latent_dim

        self.tau = args.tau
        self.encoder_tau = args.encoder_tau

        self.policy_delay = args.policy_delay
        self.actor_noise = args.actor_noise
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip

        self.training_start = args.training_start
        self.training_step = args.training_step

        self.actor = Policy_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_actor = Policy_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)
        self.target_critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_dim, activation=args.activation)

        self.encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, activation=args.activation)
        self.target_encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=args.layer_num, filter_num=args.filter_num,
                                    kernel_size=args.kernel_size, strides=args.strides, activation=args.activation)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.curl = CURL(z_dim=self.feature_dim, batch_size=self.curl_latent_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        #self.critic_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.cpc_optimizer = tf.keras.optimizers.Adam(args.cpc_lr)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2, 'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2,
                             'Curl':self.curl, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder}

        self.name = 'CURL_TD3'

    def get_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        noise = np.random.normal(loc=0, scale=self.actor_noise, size=self.action_dim)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0] + noise
        action = np.clip(action, -1, 1)

        return action

    def eval_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0]
        action = np.clip(action, -1, 1)

        return action

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_cpc_loss = 0

        for i in range(training_num):
            self.current_step += 1

            s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size)

            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]

            #curl and encoder training
            with tf.GradientTape(persistent=True) as tape:
                z_a = self.encoder(obs_anchor)
                z_pos = tf.stop_gradient(self.target_encoder(obs_pos))
                logits = self.curl.compute_logits(z_a, z_pos)
                labels = tf.range(logits.shape[0], dtype='int64')

                cpc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

            cpc_gradients = tape.gradient(cpc_loss, self.curl.trainable_variables)
            self.cpc_optimizer.apply_gradients(zip(cpc_gradients, self.curl.trainable_variables))

            encoder_gradients = tape.gradient(cpc_loss, self.encoder.trainable_variables)
            self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))

            total_cpc_loss += cpc_loss.numpy()

            del tape

            #critic network training
            target_action = tf.clip_by_value(self.target_actor(self.target_encoder(ns)) + tf.clip_by_value(
                tf.random.normal(shape=self.target_actor(self.target_encoder(ns)).shape, mean=0, stddev=self.target_noise), -self.noise_clip,
                self.noise_clip), -1, 1)

            target_value = tf.stop_gradient(
                r + self.gamma * (1 - d) * tf.minimum(self.target_critic1(self.target_encoder(ns), target_action),
                                                      self.target_critic2(self.target_encoder(ns), target_action)))

            with tf.GradientTape(persistent=True) as tape2:
                critic1_loss = tf.reduce_mean(tf.square(target_value - self.critic1(self.encoder(s), a)))
                critic2_loss = tf.reduce_mean(tf.square(target_value - self.critic2(self.encoder(s), a)))

                #critic_loss = critic1_loss + critic2_loss

            critic1_grad = tape2.gradient(critic1_loss, self.encoder.trainable_variables + self.critic1.trainable_variables)
            critic2_grad = tape2.gradient(critic2_loss, self.encoder.trainable_variables + self.critic2.trainable_variables)

            self.critic1_optimizer.apply_gradients(zip(critic1_grad, self.encoder.trainable_variables + self.critic1.trainable_variables))
            self.critic2_optimizer.apply_gradients(zip(critic2_grad, self.encoder.trainable_variables + self.critic2.trainable_variables))

            # critic_grad = tape2.gradient(critic_loss, self.encoder.trainable_variables + self.critic1.trainable_variables
            #                              + self.critic2.trainable_variables)
            # self.critic_optimizer.apply_gradients(zip(critic_grad, self.encoder.trainable_variables + self.critic1.trainable_variables
            #                                           + self.critic2.trainable_variables))

            total_c1_loss += critic1_loss.numpy()
            total_c2_loss += critic2_loss.numpy()

            del tape2

            #actor network training
            if self.current_step % self.policy_delay == 0:
                with tf.GradientTape() as tape3:
                    actor_loss = -tf.reduce_mean(self.critic1(tf.stop_gradient(self.encoder(s)), self.actor(tf.stop_gradient(self.encoder(s)))))

                actor_grad = tape3.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

                soft_update(self.actor, self.target_actor, self.tau)
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)
                soft_update(self.encoder, self.target_encoder, self.encoder_tau)

                total_a_loss += actor_loss.numpy()

                del tape3

        return {'Loss': {'Actor': total_a_loss, 'Critic1': total_c1_loss, 'Critic2': total_c2_loss, 'CPC': total_cpc_loss}}



