import tensorflow as tf
import numpy as np

class Dueling_Network(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), activation='relu',
                 kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Dueling_Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), name='Input')
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, name='Layer{}'.format(i)))

        self.value = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                           name='Value')
        self.advantage = tf.keras.layers.Dense(self.action_dim, kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer, name='Advantage')

    @tf.function
    def call(self, inputs, estimator='mean'):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)

        advantage = self.advantage(z)
        value = self.value(z)

        if estimator == 'mean':
            output = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        elif estimator == 'max':
            output = value + (advantage - tf.reduce_max(advantage, axis=1, keepdims=True))

        return output


