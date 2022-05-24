import tensorflow as tf
import numpy as np

class Dueling_Network(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):

        super(Dueling_Network, self).__init__()

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

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), name='Input')

        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(units=hidden_units[i], activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Dense{}'.format(i)))

        self.value = tf.keras.layers.Dense(units=1, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Value')

        self.advantage = tf.keras.layers.Dense(units=action_dim, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Advantage')



        self(tf.zeros(shape=(1,) + (self.state_dim,), dtype=tf.float32))

    @tf.function
    def call(self, inputs, estimator='mean'):

        assert estimator in {'mean', 'max'}

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


