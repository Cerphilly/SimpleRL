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
    def __init__(self, obs_dim, action_dim, hidden_dim=512, gamma=0.99, alpha=0.1, learning_rate=0.001, batch_size=128, buffer_size=1e6,
                 feature_dim=50, curl_latent_dim=128, layer_num=4, filter_num=32, tau=0.01, encoder_tau=0.05, training_start=1000):

        self.buffer = Buffer(buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]


        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.curl_latent_dim = curl_latent_dim

        self.layer_num = layer_num
        self.filter_num = filter_num
        self.tau = tau
        self.encoder_tau = encoder_tau

        self.training_start = training_start

        self.encoder = PixelEncoder(self.obs_dim, feature_dim, layer_num, filter_num)
        self.target_encoder = PixelEncoder(self.obs_dim, feature_dim, layer_num, filter_num)

        self.actor = Squashed_Gaussian_Actor(feature_dim, action_dim, (hidden_dim, hidden_dim), kernel_initializer=tf.keras.initializers.orthogonal())
        self.critic1 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim), kernel_initializer=tf.keras.initializers.orthogonal())
        self.critic2 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim), kernel_initializer=tf.keras.initializers.orthogonal())
        self.v_network = V_network(feature_dim, (hidden_dim, hidden_dim), kernel_initializer=tf.keras.initializers.orthogonal())
        self.target_v_network = V_network(feature_dim, (hidden_dim, hidden_dim), kernel_initializer=tf.keras.initializers.orthogonal())

        self.curl = CURL(feature_dim, self.curl_latent_dim)

        copy_weight(self.v_network, self.target_v_network)
        copy_weight(self.encoder, self.target_encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.v_network_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.cpc_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.name = 'CURL_SACv1'

    def get_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0]

        return action

    def train(self, local_step):

        s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size)

        obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]

        if local_step % 2 == 0:

            min_aq = tf.minimum(self.critic1(self.encoder(s), self.actor(self.encoder(s))),
                                self.critic2(self.encoder(s), self.actor(self.encoder(s))))
            target_v = tf.stop_gradient(min_aq - self.alpha * self.actor.log_pi(self.encoder(s)))

            with tf.GradientTape() as tape1:
                v_loss = 0.5 * tf.reduce_mean(tf.square(self.v_network(tf.stop_gradient(self.encoder(s))) - target_v))

            v_gradients = tape1.gradient(v_loss, self.v_network.trainable_variables)
            self.v_network_optimizer.apply_gradients(zip(v_gradients, self.v_network.trainable_variables))

            target_q = tf.stop_gradient(r + self.gamma * (1 - d) * self.target_v_network(self.target_encoder(ns)))

            with tf.GradientTape() as tape2:
                critic1_loss = 0.5 * tf.reduce_mean(tf.square(self.critic1(self.encoder(s), a) - target_q))

            critic1_gradients = tape2.gradient(critic1_loss,
                                               self.encoder.trainable_variables + self.critic1.trainable_variables)
            self.critic1_optimizer.apply_gradients(
                zip(critic1_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables))

            with tf.GradientTape() as tape3:
                critic2_loss = 0.5 * tf.reduce_mean(tf.square(self.critic2(self.encoder(s), a) - target_q))

            critic2_gradients = tape3.gradient(critic2_loss,
                                               self.encoder.trainable_variables + self.critic2.trainable_variables)
            self.critic2_optimizer.apply_gradients(
                zip(critic2_gradients, self.encoder.trainable_variables + self.critic2.trainable_variables))

            with tf.GradientTape() as tape4:
                mu, sigma = self.actor.mu_sigma(tf.stop_gradient(self.encoder(s)))
                output = mu + tf.random.normal(shape=mu.shape) * sigma

                min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), output),
                                        self.critic2(tf.stop_gradient(self.encoder(s)), output))

                actor_loss = tf.reduce_mean(self.alpha * self.actor.log_pi(tf.stop_gradient(self.encoder(s))) - min_aq_rep)

            actor_gradients = tape4.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            soft_update(self.v_network, self.target_v_network, self.tau)


        with tf.GradientTape(persistent=True) as tape5:
            z_a = self.encoder(obs_anchor)
            z_pos = tf.stop_gradient(self.target_encoder(obs_pos))

            logits = self.curl.compute_logits(z_a, z_pos)
            labels = tf.range(logits.shape[0], dtype='int64')

            cpc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

        cpc_gradients = tape5.gradient(cpc_loss, self.curl.trainable_variables)
        self.cpc_optimizer.apply_gradients((zip(cpc_gradients, self.curl.trainable_variables)))

        encoder_gradients = tape5.gradient(cpc_loss, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))

        soft_update(self.encoder, self.target_encoder, self.encoder_tau)

        del tape5


class CURL_SACv2:
    def __init__(self, obs_dim, action_dim, hidden_dim=1024, gamma=0.99, learning_rate=0.001, batch_size=128, buffer_size=1e6,
                 feature_dim=50, curl_latent_dim=128, layer_num=4, filter_num=32, tau=0.01, encoder_tau=0.05, training_start=1000, train_alpha=True, alpha=0.1):

        self.buffer = Buffer(buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]

        self.log_alpha = tf.Variable(initial_value=tf.math.log(alpha), trainable=True)
        self.target_entropy = -action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.curl_latent_dim = curl_latent_dim

        self.layer_num = layer_num
        self.filter_num = filter_num
        self.tau = tau
        self.encoder_tau = encoder_tau

        self.training_start = training_start
        self.train_alpha = train_alpha

        self.actor = Squashed_Gaussian_Actor(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.critic1 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.critic2 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.target_critic1 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.target_critic2 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim))

        self.encoder = PixelEncoder(self.obs_dim, feature_dim, layer_num, filter_num)
        self.target_encoder = PixelEncoder(self.obs_dim, feature_dim, layer_num, filter_num)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.curl = CURL(feature_dim, self.curl_latent_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.cpc_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(0.1 * learning_rate, beta_1=0.5)

        self.name = 'CURL_SACv2_alpha'

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0]

        return action

    def train(self, local_step):
        s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size)

        obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]

        if local_step % 2 == 0:

            target_min_aq = tf.minimum(self.target_critic1(self.target_encoder(ns), self.actor(self.encoder(ns))),
                                       self.target_critic2(self.target_encoder(ns), self.actor(self.encoder(ns))))

            target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (
                    target_min_aq - self.alpha.numpy() * self.actor.log_pi(self.encoder(ns))))

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
                mu, sigma = self.actor.mu_sigma(tf.stop_gradient(self.encoder(s)))
                output = mu + tf.random.normal(shape=mu.shape) * sigma

                min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), output),
                                        self.critic2(tf.stop_gradient(self.encoder(s)), output))

                actor_loss = tf.reduce_mean(self.alpha.numpy() * self.actor.log_pi(tf.stop_gradient(self.encoder(s))) - min_aq_rep)

            actor_gradients = tape2.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            del tape2

            if self.train_alpha == True:
                with tf.GradientTape() as tape3:
                    alpha_loss = -(tf.exp(self.log_alpha) * tf.stop_gradient(self.actor.log_pi(self.encoder(s)) + self.target_entropy))
                    alpha_loss = tf.nn.compute_average_loss(alpha_loss)

                log_alpha_gradients = tape3.gradient(alpha_loss, [self.log_alpha])
                self.log_alpha_optimizer.apply_gradients(zip(log_alpha_gradients, [self.log_alpha]))

                del tape3

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



class CURL_TD3:
    def __init__(self, obs_dim, action_dim, hidden_dim=512, gamma=0.99, learning_rate=0.001, batch_size=128, policy_delay=2, actor_noise=0.1, target_noise=0.2, noise_clip=0.5, buffer_size=1e6,
                 feature_dim=50, curl_latent_dim=128, layer_num=4, filter_num=32, tau=0.01, encoder_tau=0.05, training_start=1000):

        self.buffer = Buffer(buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = obs_dim[-1]

        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.curl_latent_dim = curl_latent_dim

        self.layer_num = layer_num
        self.filter_num = filter_num
        self.tau = tau
        self.encoder_tau = encoder_tau

        self.policy_delay = policy_delay
        self.actor_noise = actor_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.training_start = training_start

        self.actor = Policy_network(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.target_actor = Policy_network(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.critic1 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.critic2 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.target_critic1 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim))
        self.target_critic2 = Q_network(feature_dim, action_dim, (hidden_dim, hidden_dim))

        self.encoder = PixelEncoder(self.obs_dim, feature_dim, layer_num, filter_num)
        self.target_encoder = PixelEncoder(self.obs_dim, feature_dim, layer_num, filter_num)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.curl = CURL(feature_dim, self.curl_latent_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.cpc_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.name = 'CURL_TD3'

    def get_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0]

        return action

    def train(self, local_step):
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

        del tape

        if local_step % 2 == 0:
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

            if local_step % (2 * self.policy_delay) == 0:
                with tf.GradientTape() as tape2:
                    actor_loss = -tf.reduce_mean(self.critic1(tf.stop_gradient(self.encoder(s)), self.actor(tf.stop_gradient(self.encoder(s)))))

                actor_grad = tape2.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

                soft_update(self.actor, self.target_actor, self.tau)
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)
                soft_update(self.encoder, self.target_encoder, self.encoder_tau)

