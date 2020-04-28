import tensorflow as tf
import tensorflow_probability as tfp


class Gaussian_Actor(tf.keras.Model):#use it for SAC
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

        self.output_layer = tf.keras.layers.Dense(self.action_dim*2, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Output')

    @tf.function
    def call(self, input, deterministic=False):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        z = self.output_layer(z)

        mu = z[:,: self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -20.0, 2.0))

        distribution = tfp.distributions.Normal(loc=mu, scale=sigma)
        sample_action = distribution.sample()
        tanh_mean = tf.nn.tanh(mu)
        tanh_sample = tf.nn.tanh(sample_action)

        if deterministic == True:
            return tanh_mean
        else:
            return tanh_sample

    @tf.function
    def log_pi(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        z = self.output_layer(z)

        mu = z[:, : self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -20.0, 2.0))

        distribution = tfp.distributions.Normal(loc=mu, scale=sigma)
        sample_action = distribution.sample()
        tanh_sample = tf.nn.tanh(sample_action)

        log_prob = distribution.log_prob(sample_action + 1e-6)
        log_pi = log_prob - tf.reduce_sum(tf.math.log(1 - tf.square(tanh_sample) + 1e-6), axis=1, keepdims=True)
        return log_pi

    @tf.function
    def mu_sigma(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        z = self.output_layer(z)

        mu = z[:, :self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -20.0, 2.0))

        return mu, sigma






