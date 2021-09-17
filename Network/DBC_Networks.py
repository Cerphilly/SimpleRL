import tensorflow as tf
import numpy as np

class Reward_Network(tf.keras.Model):
    def __init__(self, feature_dim, hidden_dim=256, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Reward_Network, self).__init__()

        self.feature_dim = feature_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.feature_dim,), name='Input')

        self.hidden_layers = []
        self.hidden_layers.append(tf.keras.layers.Dense(hidden_dim, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Layer1"))
        self.hidden_layers.append(tf.keras.layers.LayerNormalization())
        self.hidden_layers.append(tf.keras.layers.ReLU())

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Output')

        self(tf.constant(np.zeros(shape=(1,) + (self.feature_dim,), dtype=np.float32)))


    @tf.function
    def call(self, input):

        z = self.input_layer(input)

        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)

        return output


class Transition_Network(tf.keras.Model):
    def __init__(self, feature_dim, action_dim, hidden_dim=256, deterministic = False, log_std_min=-10, log_std_max=2, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Transition_Network, self).__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.deterministic = deterministic

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.feature_dim + self.action_dim, ), name='input')

        self.hidden_layers = []
        self.hidden_layers.append(tf.keras.layers.Dense(hidden_dim, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Layer1"))
        self.hidden_layers.append(tf.keras.layers.LayerNormalization())
        self.hidden_layers.append(tf.keras.layers.ReLU())

        self.output_layers = tf.keras.layers.Dense(feature_dim * 2, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='Output')
        self(tf.constant(np.zeros(shape=(1,) + (self.feature_dim,), dtype=np.float32)), tf.constant(np.zeros(shape=(1,) + (self.action_dim,), dtype=np.float32)))

    @tf.function
    def call(self, input1, input2):
        z = tf.concat([input1, input2], axis=1)

        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layers(z)
        mean, log_std = output[:, :self.feature_dim], output[:, self.feature_dim:]
        std = tf.exp(tf.clip_by_value(log_std,  self.log_std_min, self.log_std_max))

        if self.deterministic == True:
            std = tf.zeros_like(mean)

        return mean, std

    @tf.function
    def sample(self, input1, input2):
        z = tf.concat([input1, input2], axis=1)

        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layers(z)
        mean, log_std = output[:, :self.feature_dim], output[:, self.feature_dim:]
        std = tf.exp(tf.clip_by_value(log_std, self.log_std_min, self.log_std_max))

        if self.deterministic == True:
            return mean

        else:
            eps = tf.random.normal(tf.shape(mean))
            return mean + std * eps









