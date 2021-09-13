import tensorflow as tf
import numpy as np

from Network.Gaussian_Actor import Squashed_Gaussian_Actor
from Network.Basic_Networks import Q_network, Policy_network
from Network.Encoder import PixelEncoder
from Network.DBC_Networks import Reward_Network, Transition_Network

from Common.Utils import copy_weight, soft_update
from Common.Buffer import Buffer

class DBC_SACv1:
    def __init__(self, obs_dim, action_dim, hidden_dim=512, gamma=0.99, alpha=0.1, learning_rate=0.001, batch_size=128, buffer_size=1e6,
                 feature_dim=50, curl_latent_dim=128, layer_num=4, filter_num=32, tau=0.01, encoder_tau=0.05, training_start=1000):
        pass



class DBC_SACv2:
    def __init__(self, obs_dim, action_dim, hidden_dim=256, gamma=0.99, learning_rate=1e-5, batch_size=128, buffer_size=1e6,
                 feature_dim=50, layer_num=4, filter_num=32, tau=0.005, encoder_tau=0.005, bisim_coef = 0.5, training_start=1000, train_alpha=True, alpha=0.1):

        self.buffer = Buffer(buffer_size)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.log_alpha = tf.Variable(initial_value=tf.math.log(alpha), trainable=True)
        self.target_entropy = -action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.bisim_coef = bisim_coef

        self.batch_size = batch_size
        self.feature_dim = feature_dim

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

        self.dynamics_model = Transition_Network(feature_dim, action_dim, deterministic=False)
        self.reward_model = Reward_Network(feature_dim)

        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)
        copy_weight(self.encoder, self.target_encoder)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(10*learning_rate)

        self.dynamics_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.reward_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.name = 'DBC_SACv2'

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, obs):
        obs = np.expand_dims(np.array(obs), axis=0)
        feature = self.encoder(obs)
        action = self.actor(feature).numpy()[0]

        return action

    def train(self, local_step):
        set1, set2 = self.buffer.dbc_sample(self.batch_size)

        s, a, r, ns, d = set1
        s2, a2, r2, ns2, d2 = set2

        target_min_aq = tf.minimum(self.target_critic1(self.target_encoder(ns), self.actor(self.encoder(ns))),
                                   self.target_critic2(self.target_encoder(ns), self.actor(self.encoder(ns))))

        target_q = tf.stop_gradient(r + self.gamma * (1 - d) * (target_min_aq - self.alpha.numpy() * self.actor.log_pi(self.encoder(ns))))

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

        #train dynamics(encoder used together)
        next_feature = self.encoder(ns)
        with tf.GradientTape() as tape2:
            feature = self.encoder(s)

            mu, sigma = self.dynamics_model(tf.concat([feature, a], axis=1))

            if (sigma[0][0].numpy() == 0):
                if self.dynamics_model.deterministic == False:
                    print("error")
                sigma = tf.ones_like(mu)

            diff = (mu - tf.stop_gradient(next_feature))/sigma
            dynamics_loss = tf.reduce_mean(0.5 * tf.square(diff) + tf.math.log(sigma))


        dynamics_gradients = tape2.gradient(dynamics_loss, self.encoder.trainable_variables + self.dynamics_model.trainable_variables)
        self.dynamics_optimizer.apply_gradients(zip(dynamics_gradients, self.encoder.trainable_variables + self.dynamics_model.trainable_variables))


        del tape2

        #train rewards(encoder used together)
        with tf.GradientTape() as tape3:
            feature = self.encoder(s)
            sample_dynamics = self.dynamics_model.sample(tf.concat([feature, a], axis=1))
            reward_prediction = self.reward_model(sample_dynamics)

            reward_loss = tf.reduce_mean(tf.square(reward_prediction - r))

        reward_gradients = tape3.gradient(reward_loss, self.encoder.trainable_variables + self.reward_model.trainable_variables)
        self.reward_optimizer.apply_gradients(zip(reward_gradients, self.encoder.trainable_variables + self.reward_model.trainable_variables))


        del tape3

        # train encoder
        with tf.GradientTape() as tape4:
            feature1 = self.encoder(s)
            feature2 = self.encoder(s2)

            mu1, sigma1 = self.dynamics_model(tf.concat([feature1, a], axis=1))
            mu2, sigma2 = self.dynamics_model(tf.concat([feature2, a2], axis=1))

            z_dist = tf.abs(feature1 - feature2)
            r_dist = tf.abs(r - r2)

            transition_dist = tf.sqrt(tf.square(tf.abs(mu1 - mu2)) + tf.square(tf.abs(sigma1 - sigma2)))
            bisimilarity = (tf.cast(r_dist, tf.float32) + self.gamma * tf.cast(transition_dist, tf.float32)).numpy()
            encoder_loss = self.bisim_coef * tf.reduce_mean(tf.square(z_dist - bisimilarity))

        encoder_gradients = tape4.gradient(encoder_loss, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))

        del tape4

        if local_step % 2 == 0:
            with tf.GradientTape() as tape5:
                mu, sigma = self.actor.mu_sigma(tf.stop_gradient(self.encoder(s)))
                output = mu + tf.random.normal(shape=mu.shape) * sigma

                min_aq_rep = tf.minimum(self.critic1(tf.stop_gradient(self.encoder(s)), output),
                                        self.critic2(tf.stop_gradient(self.encoder(s)), output))

                actor_loss = tf.reduce_mean(
                    self.alpha.numpy() * self.actor.log_pi(tf.stop_gradient(self.encoder(s))) - min_aq_rep)

            actor_gradients = tape5.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            del tape5

            if self.train_alpha == True:
                with tf.GradientTape() as tape6:
                    alpha_loss = -(tf.exp(self.log_alpha) * tf.stop_gradient(self.actor.log_pi(self.encoder(s)) + self.target_entropy))
                    alpha_loss = tf.nn.compute_average_loss(alpha_loss)

                log_alpha_gradients = tape6.gradient(alpha_loss, [self.log_alpha])
                self.log_alpha_optimizer.apply_gradients(zip(log_alpha_gradients, [self.log_alpha]))

                del tape6

            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)
            soft_update(self.encoder, self.target_encoder, self.encoder_tau)




class DBC_TD3:
    def __init__(self, obs_dim, action_dim, hidden_dim=512, gamma=0.99, learning_rate=0.001, batch_size=512, policy_delay=2, actor_noise=0.1, target_noise=0.2, noise_clip=0.5, buffer_size=1e6,
                 feature_dim=50, layer_num=4, filter_num=32, tau=0.005, encoder_tau=0.005, bisim_coef = 0.5, training_start=1000):

        self.buffer = Buffer(buffer_size)

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







