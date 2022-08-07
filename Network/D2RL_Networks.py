import tensorflow as tf
import tensorflow_probability as tfp

from Network.Basic_Networks import Policy_network, Q_network, V_network
from Network.Gaussian_Actor import Gaussian_Actor, Squashed_Gaussian_Actor

class D2RL_Policy(Policy_network):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256, 256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(D2RL_Policy, self).__init__(state_dim, action_dim, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    @tf.function
    def __call__(self, input, activation = 'tanh'):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)

        if activation != 'linear':
            output = tf.keras.activations.get(activation)(output)

        return output

class D2RL_Q(Q_network):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256, 256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(D2RL_Q, self).__init__(state_dim, action_dim, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    @tf.function
    def call(self, input1, input2):
        input = tf.concat([input1, input2], axis=1)

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)

        return output

class D2RL_V(V_network):
    def __init__(self, state_dim, hidden_units=(256, 256, 256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(D2RL_V, self).__init__(state_dim, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    @tf.function
    def call(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)

        return output

class D2RL_Gaussian(Gaussian_Actor):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256, 256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(D2RL_Gaussian, self).__init__(state_dim, action_dim, hidden_units, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    @tf.function
    def call(self, input, activation='tanh', deterministic=False):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)

        if activation == 'tanh':
            output = tf.nn.tanh(output)

        mean, log_std = output[:, :self.action_dim], output[:, self.action_dim:]
        std = tf.exp(log_std)

        if deterministic == True:
            return mean

        else:
            dist = tfp.distributions.Normal(loc=mean, scale=std)
            action = dist.sample()

            return action

    def dist(self, input):

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)

        output = tf.nn.tanh(output)

        mean, log_std = output[:, :self.action_dim], output[:, self.action_dim:]
        std = tf.exp(log_std)

        dist = tfp.distributions.Normal(loc=mean, scale=std)

        return dist

    def mu_sigma(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)

        output = tf.nn.tanh(output)

        mean, log_std = output[:, :self.action_dim], output[:, self.action_dim:]
        std = tf.exp(log_std)

        return mean, std

class D2RL_Squashed_Gaussian(Squashed_Gaussian_Actor):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256, 256, 256), log_std_min = -10, log_std_max=2, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(D2RL_Squashed_Gaussian, self).__init__(state_dim, action_dim, hidden_units, log_std_min, log_std_max, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)


    @tf.function
    def call(self, input, deterministic=False):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        mu = self.mean_layer(z)
        sigma = tf.exp(tf.clip_by_value(self.logstd_layer(z), self.log_std_min, self.log_std_max))

        dist = tfp.distributions.Normal(loc=mu, scale=sigma)

        if deterministic == True:
            tanh_mean = tf.nn.tanh(mu)
            log_prob = dist.log_prob(mu)
            log_pi = log_prob - tf.reduce_sum(tf.math.log(1 - tf.square(tanh_mean) + 1e-6), axis=1, keepdims=True)

            return tanh_mean, log_pi

        else:
            sample_action = dist.sample()
            tanh_sample = tf.nn.tanh(sample_action)

            log_prob = dist.log_prob(sample_action)
            log_pi = log_prob - tf.reduce_sum(tf.math.log(1 - tf.square(tanh_sample) + 1e-6), axis=1, keepdims=True)

            return tanh_sample, log_pi

    def dist(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        z = self.output_layer(z)

        mu = z[:, :self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], self.log_std_min, self.log_std_max))

        dist = tfp.distributions.Normal(loc=mu, scale=sigma)

    def mu_sigma(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        z = self.output_layer(z)

        mu = z[:, :self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], self.log_std_min, self.log_std_max))

        return mu, sigma

if __name__ == '__main__':
    import gym
    import numpy as np



    env = gym.make("Pendulum-v0")
    a = D2RL_Squashed_Gaussian(env.observation_space.shape[0], env.action_space.shape[0])
    state = env.reset()
    state = np.expand_dims(np.array(state), axis=0)
    print(a(state))
    for layer in a.get_weights():
        print(layer.shape)

    print(isinstance(a, D2RL_Gaussian))

