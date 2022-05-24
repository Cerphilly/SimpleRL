import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from Network.Basic_Network import Policy_network

class Gaussian_Policy(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), log_std_min=-10, log_std_max=2, squash=False, activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):

        super(Gaussian_Policy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden_units = hidden_units

        self.activation = activation
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.squash = squash

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), dtype=tf.float32, name='Input')
        self.hidden_layers = []

        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(units=hidden_units[i], activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Dense{}'.format(i)))

        self.mean_layer = tf.keras.layers.Dense(units=self.action_dim, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Mean')

        self.logstd_layer = tf.keras.layers.Dense(units=self.action_dim, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Logstd')

        self(tf.zeros(shape=(1,) + (self.state_dim,), dtype=tf.float32))

    @tf.function
    def call(self, input, deterministic=False):

        z = self.input_layer(input)

        for layer in self.hidden_layers:
            z = layer(z)

        mean = self.mean_layer(z)
        if self.squash == False:
            mean = tf.nn.tanh(mean)
        std = tf.exp(tf.clip_by_value(self.logstd_layer(z), self.log_std_min, self.log_std_max))

        #dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std, validate_args=True, allow_nan_stats=False)
        dist = tfp.distributions.Normal(loc=mean, scale=std, validate_args=True, allow_nan_stats=False)

        if deterministic == True:
            action = mean
        else:
            action = dist.sample()

        #log_prob = tf.reshape(dist.log_prob(action), (-1, 1))
        log_prob = dist.log_prob(action)

        if self.squash == False:
            return action, log_prob

        else:
            tanh_action = tf.nn.tanh(action)
            #log_pi = log_prob - tf.reduce_sum(tf.math.log(1 - tf.square(tanh_action) + 1e-6), axis=1, keepdims=True)

            log_pi = tf.reduce_sum(log_prob - tf.math.log(1 - tf.square(tanh_action) + 1e-6), axis=-1, keepdims=True)


            return tanh_action, log_pi

    def mu_sigma(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        mean = self.mean_layer(z)
        if self.squash == False:
            mean = tf.nn.tanh(mean)

        std = tf.exp(tf.clip_by_value(self.logstd_layer(z), self.log_std_min, self.log_std_max))

        return mean, std

    def dist(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        mean = self.mean_layer(z)
        if self.squash == False:
            mean = tf.nn.tanh(mean)

        std = tf.exp(tf.clip_by_value(self.logstd_layer(z), self.log_std_min, self.log_std_max))

        #dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std, validate_args=True, allow_nan_stats=False)
        dist = tfp.distributions.Normal(loc=mean, scale=std, validate_args=True, allow_nan_stats=False)

        return dist



if __name__ == '__main__':
    import tensorflow as tf

    a = Gaussian_Policy(5, 1)
    a(tf.zeros((2, 5)))
    print(a.hidden_units, a.log_std_max)


