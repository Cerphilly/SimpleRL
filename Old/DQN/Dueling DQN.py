import tensorflow as tf
import gym
import numpy as np
from common.ReplayBuffer import Buffer
from common.Saver import Saver

class Network(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), name='input')

        self.layer1 = tf.keras.layers.Dense(200, kernel_initializer='RandomNormal')
        self.layer2 = tf.keras.layers.Dense(200, kernel_initializer='RandomNormal')
        self.layer3 = tf.keras.layers.Dense(200, kernel_initializer='RandomNormal')

        self.value = tf.keras.layers.Dense(1, kernel_initializer='RandomNormal', name='value')
        self.advantage = tf.keras.layers.Dense(self.action_dim, kernel_initializer='RandomNormal', name='advantage')

    #@tf.function
    def call(self, inputs, estimator='mean'):
        inputs = self.input_layer(inputs)
        z = tf.nn.relu(self.layer1(inputs))
        advantage = tf.nn.relu(self.layer2(z))
        advantage = self.advantage(advantage)
        value = tf.nn.relu(self.layer3(z))
        value = self.value(value)


        if estimator == 'mean':
            output = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        elif estimator == 'max':
            output = value + (advantage - tf.reduce_max(advantage, axis=1, keepdims=True))

        return output

    def q_value(self, state):
        state = np.atleast_2d(state.astype('float32'))
        return self.predict(state)

    def best_action(self, state):
        q_value = self.q_value(state)
        return np.argmax(q_value, axis=1)


class Dueling_DQN:
    def __init__(self, state_dim, action_dim, save, load, batch_size=100, gamma=0.99, learning_rate=0.001, epsilon=0.2, training_start=200, copy_iter=5):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.save = save
        self.load = load

        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = learning_rate
        self.eps = epsilon
        self.training_start = training_start
        self.copy_iter = copy_iter


        self.network = Network(self.state_dim, self.action_dim)
        self.buffer = Buffer(self.batch_size)
        self.saver = Saver([self.network], ['network'], self.buffer, 'test_save')

        self.optimizer = tf.keras.optimizers.Adam(self.lr)


    def get_action(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(low=0, high=self.action_dim)
        else:
            return self.network.best_action(state)[0]

    def train(self, s, a, r, ns, d):

        target_q = r + self.gamma * (1 - d) * tf.reduce_max(self.network(ns), axis=1, keepdims=True)
        target_q = tf.stop_gradient(target_q)

        with tf.GradientTape() as tape:
            q_value = tf.reduce_sum(
                self.network(s) * tf.squeeze(tf.one_hot(tf.dtypes.cast(a, tf.int32), self.action_dim), axis=1), axis=1,
                keepdims=True)
            loss = 0.5*tf.math.reduce_mean(tf.square(target_q - q_value))

        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def run(self):
        if self.load == True:
            self.saver.load()

        episode = 0
        total_step = 0

        while True:
            observation = env.reset()
            done = False
            episode += 1
            episode_reward = 0
            local_step = 0

            while not done:
                local_step += 1
                total_step += 1
                env.render()

                action = self.get_action(observation)
                next_observation, reward, done, _ = env.step(action)

                episode_reward += reward

                self.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if total_step > self.training_start:
                    s, a, r, ns, d = self.buffer.sample()
                    self.train(s, a, r, ns, d)

                if done:
                    print("episode: {}, reward: {},  total_step: {}".format(episode, episode_reward, total_step))
                    if self.save == True:
                        self.saver.save()


if __name__ == '__main__':

    env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")
    #env = gym.make("Acrobot-v1")
    #env = gym.make("LunarLander-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("Dueling DQN training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    parameters = {"gamma": 0.99, "epsilon": 0.2, "learning_rate": 0.001, 'training_start': 200, 'batch_size': 100, 'copy_iter':1, 'save': False, 'load': False}

    dueling_dqn = Dueling_DQN(state_dim, action_dim, parameters['save'], parameters['load'])

    dueling_dqn.run()
