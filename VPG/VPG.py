import numpy as np
import tensorflow as tf
import gym
from common.ReplayBuffer import Buffer



class Actor(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim):
        super(Actor, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state_dim,), name='input')
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(action_dim, kernel_initializer='RandomNormal', name='output')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = self.output_layer(z)
        return output


class Critic(tf.keras.Model):
    def __init__(self, state_dim, hidden_units):
        super(Critic, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state_dim,), name='input')
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer='RandomNormal', name='output')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = tf.nn.relu(layer(z))
        output = self.output_layer(z)
        return output


class VPG:
    def __init__(self, parameters):
        self.actor = Actor(state_dim, [200, 200], action_dim)
        self.critic = Critic(state_dim, [200, 200])

        self.learning_rate = parameters['learning_rate']
        self.actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.state = np.empty([0, state_dim])
        self.action = np.empty([0, action_dim])
        self.reward = np.empty([0, 1])
        self.discounted_reward = np.empty([0, 1])




    def run(self):
        total_step = 0
        done = False
        observation = env.reset()
        while True:
            while not done:
                pass







if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("DQN training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    parameters = {"learning_rate": 0.001}



