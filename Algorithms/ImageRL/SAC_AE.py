#Improving Sample Efficiency in Model-Free Reinforcement Learning from Images, Yarats et al, 2020.

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from Networks.Gaussian_Actor import Squashed_Gaussian_Actor
from Networks.Basic_Networks import Q_network
from Networks.Encoder import PixelEncoder
from Networks.Decoder import PixelDecoder

from Common.Utils import copy_weight, soft_update, preprocess_obs
from Common.Buffer import Buffer

class SACv2_AE:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = Buffer(obs_dim, action_dim, args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = args.image_size
        self.current_step = 0

        self.log_alpha = tf.Variable(initial_value=tf.math.log(args.alpha), trainable=True)
        self.target_entropy = -action_dim
        self.gamma = args.gamma

        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim

        self.layer_num = args.layer_num
        self.filter_num = args.filter_num
        self.tau = args.tau
        self.encoder_tau = args.encoder_tau
        self.actor_update = args.actor_update
        self.critic_update = args.critic_update
        self.decoder_update = args.decoder_update
        self.decoder_latent_lambda = args.decoder_latent_lambda
        self.decoder_weight_lambda = args.decoder_weight_lambda

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
        self.decoder = PixelDecoder(self.obs_dim, self.feature_dim, self.layer_num, self.filter_num)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.decoder_optimizer = tfa.optimizers.AdamW(weight_decay=self.decoder_weight_lambda, learning_rate=args.decoder_lr)

        self.log_alpha_optimizer = tf.keras.optimizers.Adam(args.alpha_lr, beta_1=0.5)


        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder, 'Decoder': self.decoder}
        self.name = 'SACv2_AE'

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, obs):
        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature)
        action = action.numpy()[0]

        return action

    def eval_action(self, obs):

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature, deterministic=True)
        action = action.numpy()[0]

        return action

    def train(self, local_step):
        self.current_step += 1

        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_alpha_loss = 0
        total_ae_loss = 0
        loss_list = []

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        ns_action, ns_logpi = self.actor(self.encoder(ns))

        target_min_aq = tf.minimum(self.target_critic1(self.target_encoder(ns), ns_action),
                                   self.target_critic2(self.target_encoder(ns), ns_action))

        target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (target_min_aq - self.alpha.numpy() * ns_logpi))
        #critic update
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

        #actor update
        if self.current_step % self.actor_update == 0:
            with tf.GradientTape() as tape2:
                s_action, s_logpi = self.actor(tf.stop_gradient(self.encoder(s)))

                min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), s_action),
                                        self.critic2(tf.stop_gradient(self.encoder(s)), s_action))

                actor_loss = tf.reduce_mean(self.alpha.numpy() * s_logpi - min_aq_rep)

            actor_gradients = tape2.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            del tape2
            #alpha update
            if self.train_alpha == True:
                with tf.GradientTape() as tape3:
                    _, s_logpi = self.actor(self.encoder(s))
                    alpha_loss = -(tf.exp(self.log_alpha) * tf.stop_gradient(s_logpi + self.target_entropy))
                    alpha_loss = tf.nn.compute_average_loss(alpha_loss)

                log_alpha_gradients = tape3.gradient(alpha_loss, [self.log_alpha])
                self.log_alpha_optimizer.apply_gradients(zip(log_alpha_gradients, [self.log_alpha]))

                del tape3

        if self.current_step % self.decoder_update == 0:
            #encoder, decoder update
            with tf.GradientTape(persistent=True) as tape4:
                feature = self.encoder(s)
                recovered_s = self.decoder(feature)
                real_s = preprocess_obs(s)

                rec_loss = tf.reduce_mean(tf.square(recovered_s - real_s))
                latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(feature), axis=1))

                ae_loss = rec_loss + self.decoder_latent_lambda * latent_loss

            encoder_gradients = tape4.gradient(ae_loss, self.encoder.trainable_variables)
            decoder_gradients = tape4.gradient(ae_loss, self.decoder.trainable_variables)

            self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
            self.decoder_optimizer.apply_gradients(zip(decoder_gradients, self.decoder.trainable_variables))

        if self.current_step % self.critic_update == 0:

            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)
            soft_update(self.encoder, self.target_encoder, self.encoder_tau)

        del tape4

        total_c1_loss += critic1_loss.numpy()
        total_c2_loss += critic2_loss.numpy()

        loss_list.append(['Loss/Critic1', total_c1_loss])
        loss_list.append(['Loss/Critic2', total_c2_loss])

        if self.current_step % self.decoder_update == 0:
            total_ae_loss += ae_loss.numpy()
            loss_list.append(['Loss/AutoEncoder', total_ae_loss])

        if self.current_step % self.actor_update == 0:
            total_a_loss += actor_loss.numpy()
            loss_list.append(['Loss/Actor', total_a_loss])
            if self.train_alpha == True:
                total_alpha_loss += alpha_loss.numpy()
                loss_list.append(['Loss/Alpha', total_alpha_loss])

        loss_list.append(['Alpha', tf.exp(self.log_alpha).numpy()])


        return loss_list