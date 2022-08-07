import tensorflow as tf
import numpy as np

class Discriminator(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(100, 100), activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.action_dim, ), name='input')

        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, name='dense{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        self(tf.zeros(shape=(1,) + (self.state_dim,), dtype=tf.float32), tf.zeros(shape=(1,) + (self.action_dim,), dtype=tf.float32))

    @tf.function
    def call(self, input1, input2):
        z = self.input_layer(tf.concat([input1, input2], axis=1))

        for layer in self.hidden_layers:
            z = layer(z)
        output =  self.output_layer(z)

        return output

    def log_reward(self, input1, input2):
        return tf.math.log(self.call(input1, input2) + 1e-6)


