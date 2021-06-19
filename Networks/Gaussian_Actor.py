import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Gaussian_Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim, ), name='Input')
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, name='Layer{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(self.action_dim * 2, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Output')


    @tf.function
    def call(self, input, activation='tanh', deterministic=False):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

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

        output = self.output_layer(z)

        output = tf.nn.tanh(output)

        mean, log_std = output[:, :self.action_dim], output[:, self.action_dim:]
        std = tf.exp(log_std)

        return mean, std


class Squashed_Gaussian_Actor(tf.keras.Model):#use it for SAC
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), log_std_min=-10, log_std_max=2, activation='relu', kernel_initializer='RandomNormal', bias_initializer='RandomNormal'):
        super(Squashed_Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim, ), name='Input')
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, name='Layer{}'.format(i)))

        self.mean_layer = tf.keras.layers.Dense(self.action_dim, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Mean')
        self.logstd_layer = tf.keras.layers.Dense(self.action_dim, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Logstd')

        self(tf.constant(np.zeros(shape=(1,) + (self.state_dim, ), dtype=np.float32)))


    @tf.function
    def call(self, input, deterministic=False):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        mu = self.mean_layer(z)
        sigma = tf.exp(tf.clip_by_value(self.logstd_layer(z), self.log_std_min, self.log_std_max))

        if deterministic == True:
            tanh_mean = tf.nn.tanh(mu)
            return tanh_mean
        else:
            dist = tfp.distributions.Normal(loc=mu, scale=sigma, validate_args=True, allow_nan_stats=False)
            sample_action = dist.sample()
            tanh_sample = tf.nn.tanh(sample_action)

            log_prob = dist.log_prob(sample_action)
            log_pi = log_prob - tf.reduce_sum(tf.math.log(1 - tf.square(tanh_sample) + 1e-6), axis=1, keepdims=True)

            return tanh_sample, log_pi


    def dist(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        mu = self.mean_layer(z)
        sigma = tf.exp(tf.clip_by_value(self.logstd_layer(z), self.log_std_min, self.log_std_max))
        dist = tfp.distributions.Normal(loc=mu, scale=sigma, validate_args=True, allow_nan_stats=False)

        return dist

    def mu_sigma(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        mu = self.mean_layer(z)
        sigma = tf.exp(tf.clip_by_value(self.logstd_layer(z), self.log_std_min, self.log_std_max))

        return mu, sigma







