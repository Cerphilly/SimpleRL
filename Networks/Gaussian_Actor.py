import tensorflow as tf
import tensorflow_probability as tfp

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
            eps = tf.random.normal(tf.shape(mean))
            action = (mean + std * eps)
            action = tf.clip_by_value(action, -1, 1).numpy()

            #dist = tfp.distributions.Normal(loc=mean, scale=std)
            #action = dist.sample()

            return action

    @tf.function
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

    @tf.function
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
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Squashed_Gaussian_Actor, self).__init__()
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

        mu = z[:,:self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -10.0, 2.0))

        if deterministic == True:
            tanh_mean = tf.nn.tanh(mu)
            return tanh_mean
        else:
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            sample_action = dist.sample()
            tanh_sample = tf.nn.tanh(sample_action)

            return tanh_sample

    @tf.function
    def log_pi(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        z = self.output_layer(z)

        mu = z[:, : self.action_dim]
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -10.0, 2.0))
        '''
        distribution = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        sample_action = distribution.sample()
        tanh_sample = tf.nn.tanh(sample_action)

        log_prob = distribution.log_prob(sample_action + 1e-6)
        log_pi = log_prob - tf.reduce_sum(tf.math.log(1 - tf.square(tanh_sample) + 1e-6), axis=1, keepdims=True)
        '''
        distribution = tfp.distributions.Normal(loc=mu, scale=sigma)
        sample_action = mu + tf.random.normal(shape=sigma.shape) * sigma
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
        sigma = tf.exp(tf.clip_by_value(z[:, self.action_dim:], -10.0, 2.0))

        return mu, sigma







