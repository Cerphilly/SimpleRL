#CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning, Srinivas et al, 2020

import tensorflow as tf
import numpy as np

from Network.Gaussian_Policy import Gaussian_Policy
from Network.Basic_Network import Q_network
from Network.Encoder import PixelEncoder
from Network.CURL import CURL

from Common.Utils import copy_weight, soft_update, modify_default, remove_argument, modify_choices
from Common.Image_Augmentation import center_crop, random_crop
from Common.Buffer import Buffer

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


class CURL_SACv2:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = CURLBuffer(state_dim=(obs_dim[0], args.pre_image_size, args.pre_image_size) if args.data_format == 'channels_first' else (args.pre_image_size, args.pre_image_size, obs_dim[-1]),
                             action_dim=action_dim, max_size=args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.image_size = args.image_size
        self.current_step = 0

        self.log_alpha = tf.Variable(initial_value=tf.math.log(args.alpha), trainable=args.train_alpha)
        self.target_entropy = -action_dim
        self.gamma = args.gamma

        self.data_format = args.data_format

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

        self.actor = Gaussian_Policy(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_units, log_std_min=args.log_std_min, log_std_max=args.log_std_max, squash=True,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.target_critic1 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.target_critic2 = Q_network(state_dim=self.feature_dim, action_dim=self.action_dim, hidden_units=args.hidden_units,
                                      activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)

        self.encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=self.layer_num, filter_num=self.filter_num, data_format=args.data_format,
                                    activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)
        self.target_encoder = PixelEncoder(obs_dim=self.obs_dim, feature_dim=self.feature_dim, layer_num=self.layer_num, filter_num=self.filter_num, data_format=args.data_format,
                                    activation=args.activation, use_bias=args.use_bias, kernel_initializer=args.kernel_initializer, bias_initializer=args.bias_initializer)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.curl = CURL(z_dim=self.feature_dim, batch_size=self.curl_latent_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.cpc_optimizer = tf.keras.optimizers.Adam(args.cpc_lr)
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(args.alpha_lr, beta_1=0.5)

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2, 'Curl':self.curl, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder}

        self.name = 'CURL_SACv2'

    @staticmethod
    def get_config(parser):
        parser.add_argument('--log_std_min', default=-10, type=int, help='For gaussian actor')
        parser.add_argument('--log_std_max', default=2, type=int, help='For gaussian actor')
        parser.add_argument('--tau', default=0.005, type=float, help='Network soft update rate')
        parser.add_argument('--alpha', default=0.2, type=float)
        parser.add_argument('--train-alpha', default=True, type=bool)
        parser.add_argument('--alpha-lr', default=0.0005, type=float)

        parser.add_argument('--curl-latent-dim', default=128, type=int)
        parser.add_argument('--encoder-lr', default=0.001, type=float)
        parser.add_argument('--cpc-lr', default=0.001, type=float)

        parser.add_argument('--critic-update', default=2, type=int)
        parser.add_argument('--pre-image-size', default=100, type=int)

        remove_argument(parser, ['learning_rate', 'v_lr'])
        modify_choices(parser, 'train_mode', ['offline', 'online'])

        return parser

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size, self.data_format)

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature)
        action = action.numpy()[0]

        return action

    def eval_action(self, obs):

        if obs.shape[-1] != self.image_size:
            obs = center_crop(obs, self.image_size, self.data_format)

        obs = np.expand_dims(np.array(obs, dtype=np.float32), axis=0)
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

        s, a, r, ns, d, cpc_kwargs = self.buffer.cpc_sample(self.batch_size, self.image_size, self.data_format)

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
                alpha_loss = -(tf.exp(self.log_alpha) * tf.stop_gradient(s_logpi + self.target_entropy))
                alpha_loss = tf.nn.compute_average_loss(alpha_loss)
                #alpha_loss = tf.reduce_mean(alpha_loss)

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





