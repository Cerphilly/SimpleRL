#Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018.
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import cv2

import gym
import dm_control2gym
from dm_control import suite

from common.ReplayBuffer import Buffer
from common.Saver import Saver
from common.dm2gym import dmstep, dmstate, dmextendstate, dmextendstep

class Q_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim, skill_num):
        super(Q_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_num = skill_num

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.skill_num + self.action_dim,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer='RandomNormal', name='output')

    @tf.function
    def call(self, input):

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = self.output_layer(z)

        return output


class V_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, skill_num):
        super(V_network, self).__init__()
        self.state_dim = state_dim
        self.skill_num = skill_num

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.skill_num ,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer='RandomNormal', name='output')

    @tf.function
    def call(self, input):

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = self.output_layer(z)

        return output


class Policy_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim, skill_num):
        super(Policy_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_num = skill_num

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.skill_num,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(self.action_dim*2, name="output")

    @tf.function
    def call(self, input, deterministic=False):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        z = self.output_layer(z)

        mu = z[:, :self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -20.0, 2.0))

        distribution = tfp.distributions.Normal(loc=mu, scale=sigma)
        sample_action = distribution.sample()
        tanh_mean = tf.nn.tanh(mu)
        tanh_sample = tf.nn.tanh(sample_action)

        if deterministic == False:
            return tanh_sample
        else:
            return tanh_mean

    @tf.function
    def log_pi(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        z = self.output_layer(z)

        mu = z[:, :self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -20.0, 2.0))
        #print(mu, sigma)

        distribution = tfp.distributions.Normal(loc=mu, scale=sigma)  # tfp.distributions.Normal: don't get numpy array or tensor as input. use list instead. output is tensor.
        sample_action = distribution.sample()
        tanh_sample = tf.nn.tanh(sample_action)
        log_prob = distribution.log_prob(sample_action+1e-6)  # add 1e-6 to avoid -inf
        #log_prob = distribution.log_prob(sample_action)
        log_pi = log_prob - tf.reshape(tf.reduce_sum(tf.math.log(1e-6 + 1 - tf.square(tanh_sample)), axis=1),[-1,1])
        #log_pi = log_prob - tf.math.log(1e-6 + 1 - tf.square(tanh_sample))

        return log_pi

    @tf.function
    def mu_sigma(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        z = self.output_layer(z)

        mu = z[:,:self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -20.0, 2.0))

        return mu, sigma


class Discriminator(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim, skill_num, use_action=True):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_num = skill_num

        if use_action:
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.action_dim,), name='input')

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(self.skill_num, kernel_initializer='RandomNormal', name='output')

    @tf.function
    def call(self, input):

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = self.output_layer(z)

        return output



class SAC_DIAYN:
    def __init__(self, state_dim, action_dim, max_action, min_action, save, load, skill_num, batch_size=100, tau=0.995, learning_rate=0.0003, gamma=0.99, alpha=0.1, reward_scale=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.save = save
        self.load = load
        self.skill_num = skill_num
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.alpha = alpha
        self.reward_scale = reward_scale

        self.actor = Policy_network(self.state_dim, [300, 300], self.action_dim, self.skill_num)
        self.critic1 = Q_network(self.state_dim, [300, 300], self.action_dim, self.skill_num)
        self.critic2 = Q_network(self.state_dim, [300, 300], self.action_dim, self.skill_num)
        self.v_network = V_network(self.state_dim, [300, 300], self.skill_num)
        self.target_v_network = V_network(self.state_dim, [300, 300], self.skill_num)
        self.discriminator = Discriminator(self.state_dim, [100, 100], self.action_dim, self.skill_num, use_action=True)
        self.buffer = Buffer(100)

        self.actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.v_network_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.learning_rate)


        self.p_z = np.full(self.skill_num, 1.0/self.skill_num)
        self.copy_weight(self.v_network, self.target_v_network)


    def copy_weight(self, network, target_network):
        variable1 = network.weights
        variable2 = target_network.weights

        for v1, v2 in zip(variable1, variable2):
            v2.assign(v1)

    def soft_update(self, network, target_network):
        variable1 = network.weights
        variable2 = target_network.weights

        for v1, v2 in zip(variable1, variable2):
            update = self.tau * v2 + (1 - self.tau) * v1
            v2.assign(update)

    def sample_z(self):
        z = np.random.choice(self.skill_num, p=self.p_z)
        z_one_hot = np.zeros(self.skill_num)
        z_one_hot[z] = 1
        return z, z_one_hot


    def split_obs(self, obs):
        return tf.split(obs, [self.state_dim, self.skill_num], 1)

    def train(self, s, a, r, ns, d):
        with tf.GradientTape(persistent=True) as tape:

            obs, z_one_hot = self.split_obs(s)

            min_aq = tf.minimum(self.critic1(tf.concat([s, self.actor(s)], axis=1)),
                                self.critic2(tf.concat([s, self.actor(s)], axis=1)))  ###

            target_v = tf.stop_gradient(min_aq - self.alpha * self.actor.log_pi(s))
            v_loss = tf.reduce_mean(0.5 * tf.square(self.v_network(s) - target_v))

            logits = self.discriminator(obs)
            #logits = self.discriminator(tf.concat([obs, a], axis=1))

            #ns, _ = self.split_obs(ns)


            reward = -tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits(labels=z_one_hot, logits=logits), axis=1)

            #reward = tf.reduce_sum(logits*z_one_hot, axis=1, keepdims=True)
            p_z = tf.reduce_sum(self.p_z*z_one_hot, axis=1, keepdims=True)
            reward = reward - tf.math.log(p_z)

            target_q = tf.stop_gradient(reward + self.gamma * (1 - d) * self.target_v_network(ns))

            critic1_loss = 0.5 * tf.reduce_mean(tf.square(self.critic1(tf.concat([s, a], axis=1)) - (target_q)))
            critic2_loss = 0.5 * tf.reduce_mean(tf.square(self.critic2(tf.concat([s, a], axis=1)) - (target_q)))

            mu, sigma = self.actor.mu_sigma(s)

            output = mu + tf.random.normal(shape=mu.shape) * sigma

            #min_aq_rep = tf.minimum(self.critic1(tf.concat([s, output], axis=1)),
            #                       self.critic2(tf.concat([s, output], axis=1)))
            min_aq_rep = self.critic1(tf.concat([s, output], axis=1))

            # does reparametrization improves the performance of SAC? not sure. change min_aq_rep to min_aq to disable reparametrization
            actor_loss = tf.reduce_mean((self.alpha * self.actor.log_pi(s) - min_aq_rep))


            discriminator_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=z_one_hot, logits=logits))


        v_network_variables= self.v_network.trainable_variables
        v_gradients = tape.gradient(v_loss, v_network_variables)
        self.v_network_optimizer.apply_gradients(zip(v_gradients, v_network_variables))

        self.soft_update(self.v_network, self.target_v_network)

        critic1_variables = self.critic1.trainable_variables
        critic1_gradients = tape.gradient(critic1_loss, critic1_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_gradients, critic1_variables))

        critic2_variables = self.critic2.trainable_variables
        critic2_gradients = tape.gradient(critic2_loss, critic2_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_gradients, critic2_variables))

        actor_variables = self.actor.trainable_variables
        actor_grad = tape.gradient(actor_loss, actor_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, actor_variables))

        discriminator_variables = self.discriminator.trainable_variables
        discriminator_gradients = tape.gradient(discriminator_loss, discriminator_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_variables))

        del tape

    def run(self):
        episode = 0
        total_step = 0

        while True:
            episode += 1
            episode_reward = 0
            local_step = 0

            z, z_one_hot = self.sample_z()#example: 1, [0, 1]

            done = False
            observation = env.reset()
            observation = np.hstack((observation, z_one_hot))

            while not done:
                local_step += 1
                total_step += 1
                env.render()

                action = np.max(self.actor.predict(np.expand_dims(observation, axis=0).astype('float32')), axis=1)#[[  ]]줌형태로 나오므로 np.max로 []로 바꿔

                if total_step <= 5 * self.batch_size:
                    action = env.action_space.sample()

                next_observation, reward, done, _ = env.step(self.max_action * action)
                episode_reward += reward

                next_observation = np.hstack((next_observation, z_one_hot))

                self.buffer.add(observation, action, self.reward_scale * reward, next_observation, done)
                observation = next_observation


            print("episode: {}, skill: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, z, total_step, local_step,
                                                                                     episode_reward))

            if total_step >= 5 * self.batch_size:
                for i in range(local_step):
                    s, a, r, ns, d = self.buffer.sample()
                    self.train(s, a, r, ns, d)


if __name__ == '__main__':

    #env = gym.make("MountainCarContinuous-v0")

    env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    #env = gym.make("InvertedPendulumSwing-v2")#around 10000 steps.
    #env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("SAC training of", env.unwrapped.spec.id)
    sac = SAC_DIAYN(state_dim, action_dim, max_action, min_action, False, False, 3)
    sac.run()




