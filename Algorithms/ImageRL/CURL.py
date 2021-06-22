#CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning, Srinivas et al, 2020

import tensorflow as tf
import numpy as np

from Networks.Gaussian_Actor import Squashed_Gaussian_Actor
from Networks.Basic_Networks import Q_network, V_network, Policy_network
from Networks.Encoder import PixelEncoder
from Networks.CURL import CURL

from Common.Utils import copy_weight, soft_update, center_crop_image
from Common.Buffer import Buffer

class CURL_SACv1:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = Buffer(args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]

        self.gamma = args.gamma
        self.alpha = args.alpha

        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim
        self.curl_latent_dim = args.curl_latent_dim

        self.layer_num = args.layer_num
        self.filter_num = args.filter_num
        self.tau = args.tau
        self.encoder_tau = args.encoder_tau

        self.training_start = args.training_start
        self.training_step = args.training_step

        self.encoder = PixelEncoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num)
        self.target_encoder = PixelEncoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num)

        self.actor = Squashed_Gaussian_Actor(self.feature_dim, self.action_dim, args.hidden_dim, args.log_std_min, args.log_std_max, kernel_initializer=tf.keras.initializers.orthogonal())
        self.critic1 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim, kernel_initializer=tf.keras.initializers.orthogonal())
        self.critic2 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim, kernel_initializer=tf.keras.initializers.orthogonal())
        self.v_network = V_network(self.feature_dim, args.hidden_dim, kernel_initializer=tf.keras.initializers.orthogonal())
        self.target_v_network = V_network(self.feature_dim, args.hidden_dim, kernel_initializer=tf.keras.initializers.orthogonal())

        self.curl = CURL(self.feature_dim, self.curl_latent_dim)

        copy_weight(self.v_network, self.target_v_network)
        copy_weight(self.encoder, self.target_encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.v_network_optimizer = tf.keras.optimizers.Adam(args.v_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.cpc_optimizer = tf.keras.optimizers.Adam(args.cpc_lr)

        self.current_step = 0

        self.name = 'CURL_SACv1'

    def get_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature)
        action = action.numpy()[0]

        return action

    def eval_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature, deterministic=True)
        action = action.numpy()[0]

        return action

    def train(self, training_step):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_v_loss = 0
        total_cpc_loss = 0
        loss_list = []

        self.current_step += 1

        s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size)

        obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]

        s_action, s_logpi = self.actor(self.encoder(s))

        min_aq = tf.minimum(self.critic1(self.encoder(s), s_action),
                            self.critic2(self.encoder(s), s_action))
        target_v = tf.stop_gradient(min_aq - self.alpha * s_logpi)

        with tf.GradientTape() as tape1:
            v_loss = 0.5 * tf.reduce_mean(tf.square(self.v_network(tf.stop_gradient(self.encoder(s))) - target_v))

        v_gradients = tape1.gradient(v_loss, self.v_network.trainable_variables)
        self.v_network_optimizer.apply_gradients(zip(v_gradients, self.v_network.trainable_variables))

        del tape1

        target_q = tf.stop_gradient(r + self.gamma * (1 - d) * self.target_v_network(self.target_encoder(ns)))

        with tf.GradientTape(persistent=True) as tape2:
            critic1_loss = 0.5 * tf.reduce_mean(tf.square(self.critic1(self.encoder(s), a) - target_q))
            critic2_loss = 0.5 * tf.reduce_mean(tf.square(self.critic2(self.encoder(s), a) - target_q))

        critic1_gradients = tape2.gradient(critic1_loss,
                                           self.encoder.trainable_variables + self.critic1.trainable_variables)

        critic2_gradients = tape2.gradient(critic2_loss,
                                           self.encoder.trainable_variables + self.critic2.trainable_variables)

        self.critic1_optimizer.apply_gradients(
            zip(critic1_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables))

        self.critic2_optimizer.apply_gradients(
            zip(critic2_gradients, self.encoder.trainable_variables + self.critic2.trainable_variables))

        del tape2

        with tf.GradientTape() as tape3:
            s_action, s_logpi = self.actor(tf.stop_gradient(self.encoder(s)))

            min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), s_action),
                                    self.critic2(tf.stop_gradient(self.encoder(s)), s_action))

            actor_loss = tf.reduce_mean(self.alpha * s_logpi - min_aq_rep)

        actor_gradients = tape3.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        soft_update(self.v_network, self.target_v_network, self.tau)


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

        soft_update(self.encoder, self.target_encoder, self.encoder_tau)

        del tape4


        total_v_loss += v_loss.numpy()
        loss_list.append(['Loss/V', total_v_loss])

        total_c1_loss += critic1_loss.numpy()
        total_c2_loss += critic2_loss.numpy()

        loss_list.append(['Loss/Critic1', total_c1_loss])
        loss_list.append(['Loss/Critic2', total_c2_loss])

        total_a_loss += actor_loss.numpy()
        loss_list.append(['Loss/Actor', total_a_loss])

        total_cpc_loss += cpc_loss.numpy()
        loss_list.append(['Loss/CPC', total_cpc_loss])


        return loss_list



class CURL_SACv2:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = Buffer(args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]
        self.current_step = 0

        self.log_alpha = tf.Variable(initial_value=tf.math.log(args.alpha), trainable=True)
        self.target_entropy = -action_dim
        self.gamma = args.gamma

        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim
        self.curl_latent_dim = args.curl_latent_dim

        self.layer_num = args.layer_num
        self.filter_num = args.filter_num
        self.tau = args.tau
        self.encoder_tau = args.encoder_tau
        self.critic_update = args.critic_update

        self.training_start = args.training_start
        self.training_step = args.training_step
        self.train_alpha = args.train_alpha

        self.actor = Squashed_Gaussian_Actor(self.feature_dim, self.action_dim, args.hidden_dim, args.log_std_min, args.log_std_max)
        self.critic1 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim)
        self.critic2 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim)
        self.target_critic1 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim)
        self.target_critic2 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim)

        self.encoder = PixelEncoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num)
        self.target_encoder = PixelEncoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.curl = CURL(self.feature_dim, self.curl_latent_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.cpc_optimizer = tf.keras.optimizers.Adam(args.cpc_lr)
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(args.alpha_lr, beta_1=0.5)

        self.name = 'CURL_SACv2'

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature)
        action = action.numpy()[0]

        return action

    def eval_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature, deterministic=True)
        action = action.numpy()[0]

        return action

    def train(self, local_step):
        self.current_step += 1

        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_cpc_loss = 0
        total_alpha_loss = 0
        loss_list = []

        s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size)

        obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]

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
        self.critic1_optimizer.apply_gradients(
            zip(critic1_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables))

        critic2_gradients = tape1.gradient(critic2_loss,
                                           self.encoder.trainable_variables + self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(
            zip(critic2_gradients, self.encoder.trainable_variables + self.critic2.trainable_variables))

        del tape1

        with tf.GradientTape() as tape2:

            s_action, s_logpi = self.actor(tf.stop_gradient(self.encoder(s)))

            min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), s_action),
                                    self.critic2(tf.stop_gradient(self.encoder(s)), s_action))

            actor_loss = tf.reduce_mean(self.alpha.numpy() * s_logpi - min_aq_rep)

        actor_gradients = tape2.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        del tape2

        if self.train_alpha == True:
            with tf.GradientTape() as tape3:
                _, s_logpi = self.actor(self.encoder(s))
                alpha_loss = -(tf.exp(self.log_alpha) * tf.stop_gradient(s_logpi+ self.target_entropy))
                alpha_loss = tf.nn.compute_average_loss(alpha_loss)

            log_alpha_gradients = tape3.gradient(alpha_loss, [self.log_alpha])
            self.log_alpha_optimizer.apply_gradients(zip(log_alpha_gradients, [self.log_alpha]))

            del tape3


        if self.current_step % self.critic_update == 0:
            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)

        soft_update(self.encoder, self.target_encoder, self.encoder_tau)


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

        del tape4


        total_c1_loss += critic1_loss.numpy()
        total_c2_loss += critic2_loss.numpy()

        loss_list.append(['Loss/Critic1', total_c1_loss])
        loss_list.append(['Loss/Critic2', total_c2_loss])

        total_a_loss += actor_loss.numpy()
        loss_list.append(['Loss/Actor', total_a_loss])

        total_cpc_loss += cpc_loss.numpy()
        loss_list.append(['Loss/CPC', total_cpc_loss])

        if self.train_alpha == True:
            total_alpha_loss += alpha_loss.numpy()
            loss_list.append(['Loss/Alpha', total_alpha_loss])

        loss_list.append(['Alpha', tf.exp(self.log_alpha).numpy()])

        return loss_list


class CURL_TD3:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = Buffer(args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]

        self.current_step = 0

        self.gamma = args.gamma

        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim
        self.curl_latent_dim = args.curl_latent_dim

        self.layer_num = args.layer_num
        self.filter_num = args.filter_num
        self.tau = args.tau
        self.encoder_tau = args.encoder_tau

        self.policy_delay = args.policy_delay
        self.actor_noise = args.actor_noise
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip

        self.training_start = args.training_start
        self.training_step = args.training_step

        self.actor = Policy_network(self.feature_dim, self.action_dim, args.hidden_dim)
        self.target_actor = Policy_network(self.feature_dim, self.action_dim, args.hidden_dim)
        self.critic1 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim)
        self.critic2 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim)
        self.target_critic1 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim)
        self.target_critic2 = Q_network(self.feature_dim, self.action_dim, args.hidden_dim)

        self.encoder = PixelEncoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num)
        self.target_encoder = PixelEncoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.curl = CURL(self.feature_dim, self.curl_latent_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.cpc_optimizer = tf.keras.optimizers.Adam(args.cpc_lr)

        self.name = 'CURL_TD3'

    def get_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        noise = np.random.normal(loc=0, scale=self.actor_noise, size=self.action_dim)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0] + noise
        action = np.clip(action, -1, 1)

        return action

    def eval_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0]
        action = np.clip(action, -1, 1)

        return action

    def train(self, local_step):
        self.current_step += 1

        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_cpc_loss = 0
        loss_list = []
        s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size)

        obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]

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
        loss_list.append(['Loss/CPC', total_cpc_loss])

        del tape

        if self.current_step % 2 == 0:
            target_action = tf.clip_by_value(self.target_actor(self.target_encoder(ns)) + tf.clip_by_value(
                tf.random.normal(shape=self.target_actor(self.target_encoder(ns)).shape, mean=0, stddev=self.target_noise), -self.noise_clip,
                self.noise_clip), -1, 1)

            target_value = tf.stop_gradient(
                r + self.gamma * (1 - d) * tf.minimum(self.target_critic1(self.target_encoder(ns), target_action),
                                                      self.target_critic2(self.target_encoder(ns), target_action)))

            with tf.GradientTape(persistent=True) as tape:
                critic1_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic1(self.encoder(s), a)))
                critic2_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic2(self.encoder(s), a)))

            critic1_grad = tape.gradient(critic1_loss, self.encoder.trainable_variables + self.critic1.trainable_variables)
            self.critic1_optimizer.apply_gradients(zip(critic1_grad, self.encoder.trainable_variables + self.critic1.trainable_variables))

            critic2_grad = tape.gradient(critic2_loss, self.encoder.trainable_variables + self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(critic2_grad, self.encoder.trainable_variables + self.critic2.trainable_variables))

            if self.current_step % (2 * self.policy_delay) == 0:
                with tf.GradientTape() as tape2:
                    actor_loss = -tf.reduce_mean(self.critic1(tf.stop_gradient(self.encoder(s)), self.actor(tf.stop_gradient(self.encoder(s)))))

                actor_grad = tape2.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

                soft_update(self.actor, self.target_actor, self.tau)
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)
                soft_update(self.encoder, self.target_encoder, self.encoder_tau)

                total_a_loss += actor_loss.numpy()
                loss_list.append(['Loss/Actor', total_a_loss])

            total_c1_loss += critic1_loss.numpy()
            total_c2_loss += critic2_loss.numpy()

            loss_list.append(['Loss/Critic1', total_c1_loss])
            loss_list.append(['Loss/Critic2', total_c2_loss])

        return loss_list



