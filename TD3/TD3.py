#Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018.
import tensorflow as tf
import gym
import numpy as np
import cv2
from dm_control import suite

from common.ReplayBuffer import Buffer
from common.Saver import Saver
from common.dm2gym import dmstate, dmstep

class Actor(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim, max_action):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(self.action_dim, kernel_initializer='RandomNormal')


    @tf.function
    def call(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = self.max_action*tf.nn.tanh(self.output_layer(z))

        return output

class Critic(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.action_dim,), name='input')

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer='RandomNormal')

    @tf.function
    def call(self, input):

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = self.output_layer(z)

        return output

class TD3:
    def __init__(self, state_dim, action_dim, max_action, min_action, save, load, batch_size=100, gamma=0.99, tau=0.995, learning_rate=0.001, policy_delay=2,
                 actor_noise=0.1, target_noise=0.2, noise_clip=0.5, training_start=1000):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.save = save
        self.load = load

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = learning_rate
        self.policy_delay = policy_delay
        self.actor_noise = actor_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.training_start = training_start

        self.total_step = 0

        self.actor = Actor(self.state_dim, [256, 256], self.action_dim, self.max_action)
        self.target_actor = Actor(self.state_dim, [256, 256], self.action_dim, self.max_action)
        self.critic1 = Critic(self.state_dim, [256, 256], self.action_dim)
        self.target_critic1 = Critic(self.state_dim, [256, 256], self.action_dim)
        self.critic2 = Critic(self.state_dim, [256, 256], self.action_dim)
        self.target_critic2 = Critic(self.state_dim, [256, 256], self.action_dim)

        self.buffer = Buffer(self.batch_size)
        self.saver = Saver([self.actor, self.target_actor, self.critic1, self.target_critic1, self.critic2, self.target_critic2],
                           ['actor', 'target_actor', 'critic1', 'target_critic1', 'critic2', 'target_critic2'], self.buffer,
                           'TD3_threepole')

        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(self.lr)


        self.copy_weight(self.actor, self.target_actor)
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
            update = self.tau*v2 + (1-self.tau)*v1
            v2.assign(update)

    def train(self, s, a, r, ns, d):

        target_action = np.clip(self.target_actor(ns).numpy() \
                                + np.clip(
            np.random.normal(loc=0, scale=self.target_noise, size=np.array([self.batch_size, 1]))
            , -self.noise_clip, self.noise_clip), self.min_action, self.max_action)

        # target_action = tf.clip_by_value(self.target_actor(ns) + tf.clip_by_value(tf.random.normal(shape=self.target_actor(ns).shape, mean=0, stddev=self.target_noise), -self.noise_clip, self.noise_clip), -max_action, max_action)

        target_value = r + self.gamma * (1 - d) * tf.minimum(
            self.target_critic1(tf.concat([ns, target_action], axis=1)),
            self.target_critic2(tf.concat([ns, target_action], axis=1)))
        target_value = tf.stop_gradient(target_value)

        with tf.GradientTape(persistent=True) as tape:
            critic1_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic1(tf.concat([s, a], axis=1))))
            critic2_loss = 0.5 * tf.reduce_mean(tf.square(target_value - self.critic2(tf.concat([s, a], axis=1))))

        critic1_variables = self.critic1.trainable_variables
        critic1_grad = tape.gradient(critic1_loss, critic1_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grad, critic1_variables))

        critic2_variables = self.critic2.trainable_variables
        critic2_grad = tape.gradient(critic2_loss, critic2_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grad, critic2_variables))

        if self.total_step % 2 == 0:
            with tf.GradientTape() as tape:
                actor_loss = -tf.reduce_mean(self.critic1(tf.concat([s, self.actor(s)], axis=1)))
            actor_variables = self.actor.trainable_variables
            actor_grad = tape.gradient(actor_loss, actor_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, actor_variables))

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)


    def run(self):
        if self.load == True:
            self.saver.load()

        episode = 0
        self.total_step = 0

        while True:
            episode += 1
            episode_reward = 0
            step = 0

            done = False
            observation = env.reset()

            while not done:
                step += 1
                self.total_step += 1
                env.render()

                noise = np.random.normal(loc=0, scale=self.actor_noise, size=self.action_dim)
                action = np.max(self.actor.predict(np.expand_dims(observation, axis=0).astype('float32')),
                                axis=1) + noise
                action = np.clip(action, self.min_action, self.max_action)

                if self.total_step < self.training_start:
                    action = env.action_space.sample()

                next_observation, reward, done, _ = env.step(action)
                episode_reward += reward
                self.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if self.total_step >= self.training_start:
                    s, a, r, ns, d = self.buffer.sample()
                    self.train(s, a, r, ns, d)

            if self.save == True:
                self.saver.save()

            print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, self.total_step, step, episode_reward))

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

                if total_step <= self.training_start:
                    action = np.random.uniform(self.min_action, self.max_action)

                next_observation, reward, done = dmstep(env.step(action))

                episode_reward += reward

                self.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                cv2.imshow('result', video[local_step - 1])
                cv2.waitKey(1)

                if local_step == 1000: done = True

            if total_step >= self.training_start:
                for _ in range(100):
                    s, a, r, ns, d = self.buffer.sample()
                    self.train(s, a, r, ns, d)



            print("episode: {}, total_step: {}, step: {}, episode_reward: {}".format(episode, total_step, local_step, episode_reward))

            if self.save == True:
                if total_step % 100000 == 0:
                    self.saver.save()

if __name__ == '__main__':

    #env = gym.make("Pendulum-v0")
    #env = gym.make("MountainCarContinuous-v0")
    env = gym.make("InvertedTriplePendulumSwing-v2")

    # env = gym.make("InvertedDoublePendulumSwing-v2")
    # env = gym.make("InvertedDoublePendulum-v2")
    #env = gym.make("InvertedPendulumSwing-v2")
    #env = gym.make("InvertedPendulum-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    
    print("TD3 training of", env.unwrapped.spec.id)
    '''

    env = suite.load(domain_name="cartpole", task_name="three_poles")
    state_spec = env.reset()
    action_spec = env.action_spec()
    state_dim = len(dmstate(state_spec))
    action_dim = action_spec.shape[0]  # 1
    max_action = action_spec.maximum[0]  # 1.0
    min_action = action_spec.minimum[0]


    parameters = {"gamma": 0.99, "tau": 0.995, "learning_rate": 0.001, "policy_delay": 2, "actor_noise": 0.1, "target_noise": 0.2,
                  "noise_clip": 0.5, "training_start": 1000, "batch_size": 100, 'save': True, 'load': True}

    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)


    td3 = TD3(state_dim, action_dim, max_action, min_action, parameters['save'], parameters['load'])
    td3.run_dm()
    #50만 학습



