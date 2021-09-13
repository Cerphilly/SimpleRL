#Learning Invariant Representations for Reinforcement Learning without Reconstruction, A. Zhang et al, 2020.

import tensorflow as tf
import numpy as np

from Network.Gaussian_Actor import Squashed_Gaussian_Actor
from Network.Basic_Networks import Q_network, Policy_network
from Network.Encoder import PixelEncoder
from Network.DBC_Networks import Reward_Network, Transition_Network

from Common.Utils import copy_weight, soft_update
from Common.Buffer import Buffer

class DBC_SACv2:
    def __init__(self, obs_dim, action_dim, args):

        self.buffer = Buffer(state_dim=obs_dim, action_dim=action_dim, max_size=args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

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

        self.dynamics_model = Transition_Network(self.feature_dim, action_dim, deterministic=False)
        self.reward_model = Reward_Network(self.feature_dim)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(args.critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(args.critic_lr)

        self.encoder_optimizer = tf.keras.optimizers.Adam(args.encoder_lr)
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(args.alpha_lr)

        self.dynamics_optimizer = tf.keras.optimizers.Adam(args.decoder_lr)
        self.reward_optimizer = tf.keras.optimizers.Adam(args.decoder_lr)

        self.current_step = 0

        self.network_list = {'Actor': self.actor, 'Critic1': self.critic1, 'Critic2': self.critic2,
                             'Target_Critic1': self.target_critic1, 'Target_Critic2': self.target_critic2, 'Encoder': self.encoder, 'Target_Encoder': self.target_encoder, 'Dynamics': self.dynamics_model, 'Reward': self.reward_model}

        self.name = 'DBC_SACv2'

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, obs):
        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature)
        action = action.numpy()[0]

        return action

    def eval_action(self, obs):
        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action, _ = self.actor(feature, deterministic=True)
        action = action.numpy()[0]

        return action

    def train(self, local_step):
        self.current_step += 1
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_alpha_loss = 0
        total_encoder_loss = 0
        total_dynamics_loss = 0
        total_reward_loss = 0
        loss_list = []
        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        ns_action, ns_logpi = self.actor(self.encoder(ns))

        target_min_aq = tf.minimum(self.target_critic1(self.target_encoder(ns), ns_action),
                                   self.target_critic2(self.target_encoder(ns), ns_action))

        target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (target_min_aq - self.alpha.numpy() * ns_logpi))

        with tf.GradientTape(persistent=True) as tape1:
            critic1_loss = tf.reduce_mean(tf.square(self.critic1(self.encoder(s), a) - target_q))
            critic2_loss = tf.reduce_mean(tf.square(self.critic2(self.encoder(s), a) - target_q))

        critic1_gradients = tape1.gradient(critic1_loss, self.encoder.trainable_variables + self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_gradients, self.encoder.trainable_variables + self.critic1.trainable_variables))

        critic2_gradients = tape1.gradient(critic2_loss, self.encoder.trainable_variables + self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_gradients, self.encoder.trainable_variables + self.critic2.trainable_variables))

        del tape1

        if self.current_step % self.actor_update == 0:
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

        #train encoder
        with tf.GradientTape() as tape4:
            new_ids = np.arange(len(s))
            np.random.shuffle(new_ids)
            s2 = tf.gather(s, new_ids)

            feature = self.encoder(s)
            #feature2 = tf.gather(feature, new_ids)
            feature2 = self.encoder(s2)

            reward = self.reward_model(tf.stop_gradient(feature))
            #reward2 = tf.gather(reward, new_ids)
            reward2 = self.reward_model(tf.stop_gradient(feature2))

            feature_action, _ = self.actor(tf.stop_gradient(feature), True)
            feature2_action, _ = self.actor(tf.stop_gradient(feature2), True)

            mu, sigma = self.dynamics_model(tf.stop_gradient(feature), feature_action)
            mu2, sigma2 = self.dynamics_model(tf.stop_gradient(feature2), feature2_action)

            z_dist = tf.reshape(tf.keras.losses.huber(feature, feature2), shape=[-1, 1])
            r_dist = tf.reshape(tf.keras.losses.huber(reward, reward2), shape=[-1, 1])
            transition_dist = tf.sqrt(tf.square(mu - mu2) + tf.square(sigma - sigma2))

            bisimilarity = r_dist + self.gamma * transition_dist
            encoder_loss = tf.reduce_mean(tf.square(z_dist - bisimilarity))

        encoder_gradients = tape4.gradient(encoder_loss, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))

        #train dynamics
        with tf.GradientTape() as tape5:
            feature = self.encoder(s)
            mu, sigma = self.dynamics_model(feature, a)

            if (sigma[0][0].numpy() == 0):
                 if self.dynamics_model.deterministic == False:
                     print("error")
                 sigma = tf.ones_like(mu)

            next_feature = self.encoder(ns)
            diff = (mu - tf.stop_gradient(next_feature)) / sigma

            dynamics_loss = tf.reduce_mean(0.5 * tf.square(diff) + tf.math.log(sigma))

        dynamics_gradients = tape5.gradient(dynamics_loss, self.encoder.trainable_variables + self.dynamics_model.trainable_variables)
        self.dynamics_optimizer.apply_gradients(zip(dynamics_gradients, self.encoder.trainable_variables + self.dynamics_model.trainable_variables))

        #train reward
        with tf.GradientTape() as tape6:
            feature = self.encoder(s)
            sample_dynamics = self.dynamics_model.sample(feature, a)
            reward_prediction = self.reward_model(sample_dynamics)

            reward_loss = tf.reduce_mean(tf.square(reward_prediction - r))

        reward_gradients = tape6.gradient(reward_loss, self.encoder.trainable_variables + self.reward_model.trainable_variables)
        self.reward_optimizer.apply_gradients(zip(reward_gradients, self.encoder.trainable_variables + self.reward_model.trainable_variables))


        total_c1_loss += critic1_loss.numpy()
        total_c2_loss += critic2_loss.numpy()

        loss_list.append(['Loss/Critic1', total_c1_loss])
        loss_list.append(['Loss/Critic2', total_c2_loss])

        if self.current_step % self.actor_update == 0:
            total_a_loss += actor_loss.numpy()
            loss_list.append(['Loss/Actor', total_a_loss])

        total_encoder_loss += encoder_loss.numpy()
        loss_list.append(['Loss/Encoder', total_encoder_loss])

        total_dynamics_loss += dynamics_loss.numpy()
        loss_list.append(['Loss/Dynamics', total_dynamics_loss])

        total_reward_loss += reward_loss.numpy()
        loss_list.append(['Loss/Reward', total_reward_loss])

        if self.current_step % self.actor_update == 0 and self.train_alpha == True:
            total_alpha_loss += alpha_loss.numpy()
            loss_list.append(['Loss/Alpha', total_alpha_loss])

        loss_list.append(['Alpha', tf.exp(self.log_alpha).numpy()])

        return loss_list





class DBC_TD3:
    def __init__(self, obs_dim, action_dim, hidden_dim=512, gamma=0.99, learning_rate=0.001, batch_size=512, policy_delay=2, actor_noise=0.1, target_noise=0.2, noise_clip=0.5, buffer_size=1e6,
                 feature_dim=50, layer_num=4, filter_num=32, tau=0.005, encoder_tau=0.005, bisim_coef = 0.5, training_start=1000):

        self.buffer = Buffer(state_dim=obs_dim, action_dim=action_dim, max_size=args.buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.feature_dim = feature_dim

        self.layer_num = layer_num
        self.filter_num = filter_num
        self.tau = tau
        self.encoder_tau = encoder_tau
        self.bisim_coef = bisim_coef

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

        self.dynamics_model = Transition_Network(feature_dim, action_dim)
        self.reward_model = Reward_Network(feature_dim)

        copy_weight(self.actor, self.target_actor)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.dynamics_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.reward_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.name = 'DBC_TD3'

    def get_action(self, obs):
        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0]

        return action

    def train(self, local_step):#critic -> transition -> reward -> encoder -> actor
        set1, set2 = self.buffer.dbc_sample(self.batch_size)

        s, a, r, ns, d = set1
        s2, a2, r2, ns2, d2 = set2

        target_action = tf.clip_by_value(self.target_actor(self.target_encoder(ns)) + tf.clip_by_value(
            tf.random.normal(shape=self.target_actor(self.target_encoder(ns)).shape, mean=0, stddev=self.target_noise), -self.noise_clip,
            self.noise_clip), -1, 1)

        target_value = tf.stop_gradient(
            r + self.gamma * (1 - d) * tf.minimum(self.target_critic1(self.target_encoder(ns), target_action),
                                                  self.target_critic2(self.target_encoder(ns), target_action)))

        with tf.GradientTape(persistent=True) as tape1:
            critic1_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic1(self.encoder(s), a)))
            critic2_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic2(self.encoder(s), a)))

        critic1_grad = tape1.gradient(critic1_loss, self.encoder.trainable_variables + self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grad, self.encoder.trainable_variables + self.critic1.trainable_variables))

        critic2_grad = tape1.gradient(critic2_loss, self.encoder.trainable_variables + self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grad, self.encoder.trainable_variables + self.critic2.trainable_variables))

        del tape1

        #train dynamics
        with tf.GradientTape() as tape2:
            feature = self.encoder(s)
            next_feature = self.encoder(ns)
            mu, sigma = self.dynamics_model(tf.concat([feature, a], axis=1))

            if (sigma[0][0].numpy() == 0):
                sigma = tf.ones_like(mu)
            diff = (mu - tf.stop_gradient(next_feature))/sigma
            dynamics_loss = tf.reduce_mean(0.5 * tf.square(diff) + tf.math.log(sigma))

        dynamics_gradients = tape2.gradient(dynamics_loss, self.encoder.trainable_variables + self.dynamics_model.trainable_variables)
        self.dynamics_optimizer.apply_gradients(zip(dynamics_gradients, self.encoder.trainable_variables + self.dynamics_model.trainable_variables))

        #dynamics_gradients = tape2.gradient(dynamics_loss, self.dynamics_model.trainable_variables)
        #self.dynamics_optimizer.apply_gradients(zip(dynamics_gradients, self.dynamics_model.trainable_variables))

        del tape2

        #train reward
        with tf.GradientTape() as tape3:
            feature = self.encoder(s)
            sample_dynamics = self.dynamics_model.sample(tf.concat([feature, a], axis=1))
            reward_prediction = self.reward_model(sample_dynamics)

            reward_loss = tf.reduce_mean(tf.square(reward_prediction - (r)))

        reward_gradients = tape3.gradient(reward_loss, self.encoder.trainable_variables + self.reward_model.trainable_variables)
        self.reward_optimizer.apply_gradients(zip(reward_gradients, self.encoder.trainable_variables + self.reward_model.trainable_variables))

        #reward_gradients = tape3.gradient(reward_loss, self.reward_model.trainable_variables)
        #self.reward_optimizer.apply_gradients(zip(reward_gradients, self.reward_model.trainable_variables))

        del tape3

        #train encoder
        with tf.GradientTape() as tape4:
            feature1 = self.encoder(s)
            feature2 = self.encoder(s2)

            mu1, sigma1 = self.dynamics_model(tf.concat([feature1, a], axis=1))
            mu2, sigma2 = self.dynamics_model(tf.concat([feature2, a2], axis=1))

            z_dist = tf.abs(feature1 - feature2)
            r_dist = tf.abs(r - r2)

            transition_dist = tf.sqrt(tf.square(tf.abs(mu1 - mu2)) + tf.square(tf.abs(sigma1 - sigma2)))
            bisimilarity = tf.stop_gradient(tf.cast(r_dist, tf.float32) + self.gamma * tf.cast(transition_dist, tf.float32))
            encoder_loss = self.bisim_coef * tf.reduce_mean(tf.square(z_dist - bisimilarity))

        encoder_gradients = tape4.gradient(encoder_loss, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))

        del tape4

        if local_step % (self.policy_delay) == 0:
            with tf.GradientTape() as tape5:
                actor_loss = -tf.reduce_mean(self.critic1(tf.stop_gradient(self.encoder(s)), self.actor(tf.stop_gradient(self.encoder(s)))))

            actor_grad = tape5.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            del tape5

            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)
            soft_update(self.encoder, self.target_encoder, self.encoder_tau)







