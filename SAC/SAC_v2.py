#Soft Actor-Critic Algorithms and Applications, Haarnoja et al, 2018
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym
from dm_control import suite
import cv2
from common.ReplayBuffer import Buffer
from common.Saver import Saver
from common.dm2gym import dmstep, dmstate


class Q_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim):
        super(Q_network, self).__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.action_dim,), name='input')

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
    def __init__(self, state_dim, hidden_units, action_dim):
        super(Policy_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(self.action_dim * 2, name="output")

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

        log_prob = tf.math.log(distribution.prob(sample_action) + 1e-6)  # add 1e-6 to avoid -inf
        #log_prob = distribution.log_prob(sample_action)
        #log_pi = (log_prob - tf.math.log(1e-6 + 1 - tf.square(tanh_sample)))
        log_pi = log_prob - tf.reshape(tf.reduce_sum(tf.math.log(1e-6 + 1 - tf.square(tanh_sample)), axis=1),[-1,1])


        return log_pi

    @tf.function
    def mu_sigma(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        z = self.output_layer(z)

        mu = z[:, :action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -20.0, 2.0))

        return mu, sigma

class SAC:
    def __init__(self, state_dim, action_dim, max_action, min_action, save, load, batch_size=100, alpha=0.1, tau=0.995, learning_rate=0.0003, gamma=0.99, auto_alpha=False, reward_scale=1):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.save = save
        self.load = load

        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.auto_alpha = auto_alpha
        self.reward_scale = reward_scale

        if self.auto_alpha == False:
            self.alpha = alpha
        else:
            self.log_alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha.assign(tf.exp(self.log_alpha))
            self.target_alpha = -action_dim
            self.alpha_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.actor = Policy_network(self.state_dim, [256, 256], self.action_dim)
        self.critic1 = Q_network(self.state_dim, [256, 256], self.action_dim)
        self.target_critic1 = Q_network(self.state_dim, [256, 256], self.action_dim)
        self.critic2 = Q_network(self.state_dim, [256, 256], self.action_dim)
        self.target_critic2 = Q_network(self.state_dim, [256, 256], self.action_dim)

        self.buffer = Buffer(self.batch_size)
        self.saver = Saver([self.actor, self.critic1, self.target_critic1, self.critic2, self.target_critic2], ['actor', 'critic1', 'target_critic1', 'critic2', 'target_critic2'], self.buffer, '/home/cocel/PycharmProjects/SimpleRL/TD3/TD3_threepole')

        self.actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.copy_weight(self.critic1, self.target_critic1)
        self.copy_weight(self.critic2, self.target_critic2)

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


    def train(self, s, a, r, ns, d):
        #s, a, r, ns, d = self.buffer.ERE_sample(i, update_len)


        with tf.GradientTape(persistent=True) as tape:
            target_min_aq = tf.minimum(self.target_critic1(tf.concat([ns, self.actor(ns)], axis=1)),
                                       self.target_critic2(tf.concat([ns, self.actor(ns)], axis=1)))  ###

            target_q = r + self.gamma * (1 - d) * (target_min_aq - self.alpha * self.actor.log_pi(ns))

            target_q = tf.stop_gradient(target_q)

            critic1_loss = tf.reduce_mean(tf.square(self.critic1(tf.concat([s, a], axis=1)) - target_q))

            critic2_loss = tf.reduce_mean(tf.square(self.critic2(tf.concat([s, a], axis=1)) - target_q))

            min_aq = tf.minimum(self.critic1(tf.concat([s, self.actor(s)], axis=1)),
                                self.critic2(tf.concat([s, self.actor(s)], axis=1)))

            mu, sigma = self.actor.mu_sigma(s)
            output = mu + tf.random.normal(shape=mu.shape) * sigma

            #min_aq_rep = tf.minimum(self.critic1(tf.concat([s, output], axis=1)),
            #                        self.critic2(tf.concat([s, output], axis=1)))
            min_aq_rep = self.critic1(tf.concat([s, output], axis=1))


            actor_loss = tf.reduce_mean(self.alpha * self.actor.log_pi(s) - min_aq_rep)

            if self.auto_alpha == True:
                alpha_loss = -tf.reduce_mean(self.log_alpha*tf.stop_gradient(self.actor.log_pi(s) + self.target_alpha))


        critic1_gradients = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_gradients, self.critic1.trainable_variables))
        critic2_gradients = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_gradients, self.critic2.trainable_variables))

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        if self.auto_alpha == True:
            alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)




    def run(self):
        if self.load == True:
            self.saver.load()

        episode = 0
        total_step = 0

        while True:
            episode += 1
            episode_reward = 0
            local_step = 0

            done = False
            observation = env.reset()

            while not done:
                local_step += 1
                total_step += 1
                env.render()

                action = np.max(self.actor.predict(np.expand_dims(observation, axis=0).astype('float32')), axis=1)

                if total_step <= 5 * self.batch_size:
                    action = env.action_space.sample()

                next_observation, reward, done, _ = env.step(self.max_action * action)
                episode_reward += reward

                self.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation


            if self.auto_alpha == True:
                print("episode: {}, total_step: {}, step: {}, episode_reward: {}, alpha: {}".format(episode, total_step, local_step,
                                                                                         episode_reward, self.alpha.numpy()))

            else:
                print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, total_step, local_step,
                                                                                     episode_reward))


            if total_step >= 5*self.batch_size:
                for i in range(local_step):
                    s, a, r, ns, d = self.buffer.sample()
                    self.train(s, a, r, ns, d)

            if self.save == True:
                self.saver.save()

    def run_dm(self):
        if self.load == True:
            self.saver.load()

        episode = 0
        total_step = 0

        height = 480
        width = 640

        video = np.zeros((1001, height, width, 3), dtype=np.uint8)

        while True:
            episode += 1
            episode_reward = 0
            local_step = 0

            done = False
            observation = dmstate(env.reset())

            while not done:
                local_step += 1
                total_step += 1

                x = env.physics.render(height=480, width=640, camera_id=0)
                video[local_step] = x

                action = np.max(self.actor.predict(np.expand_dims(observation, axis=0).astype('float32')), axis=1)

                if total_step <= 5 * self.batch_size:
                    action = np.random.uniform(self.min_action, self.max_action)

                next_observation, reward, done = dmstep(env.step(self.max_action*action))

                episode_reward += reward

                self.buffer.add(observation, action, self.reward_scale * reward, next_observation, done)
                observation = next_observation

                cv2.imshow('result', video[local_step - 1])
                cv2.waitKey(1)
                if local_step == 1000: done = True

            if self.auto_alpha == True:
                print("episode: {}, total_step: {}, step: {}, episode_reward: {}, alpha: {}".format(episode, total_step,
                                                                                                    local_step,
                                                                                                    episode_reward,
                                                                                                    self.alpha.numpy()))

            else:
                print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, total_step, local_step,
                                                                                         episode_reward))

            if total_step >= 5 * self.batch_size:
                for i in range(100):
                    s, a, r, ns, d = self.buffer.sample()
                    self.train(s, a, r, ns, d)

            if self.save == True:
                self.saver.save()


if __name__ == '__main__':

    #env = gym.make("Pendulum-v0")#around 5000 steps
    #env = gym.make("MountainCarContinuous-v0")

    env = gym.make("InvertedTriplePendulumSwing-v2")

    # env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    # env = gym.make("InvertedPendulumSwing-v2")
    # env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("SAC training of", env.unwrapped.spec.id)
    '''
    #env = suite.load(domain_name="pendulum", task_name="swingup")
    env = suite.load(domain_name="cartpole", task_name="three_poles")
    #env = suite.load(domain_name="cartpole", task_name="swingup")
    state_spec = env.reset()
    action_spec = env.action_spec()
    state_dim = len(dmstate(state_spec))
    action_dim = action_spec.shape[0]  # 1
    max_action = action_spec.maximum[0]  # 1.0
    min_action = action_spec.minimum[0]
    '''

    parameters = {'tau': 0.995, "learning_rate": 0.01, 'gamma': 0.99, 'alpha':0.1, 'batch_size': 100, 'auto_alpha': False, 'reward_scale': 1, 'save': False, 'load': False}


    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)

    sac = SAC(state_dim, action_dim, max_action, min_action, False, False)
    sac.run()
    #TD3 1200000 step to SAC_v2