import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from Common.Utils import TanhBijector, SampleDist, OneHotDist

class DreamerConvEncoder(tf.keras.Model):
    def __init__(self, state_dim, depth=32, activation='relu'):
        super(DreamerConvEncoder, self).__init__()
        self.state_dim = state_dim
        self.depth = depth
        #self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim), name='Input')

        self.conv_layers = []
        self.conv_layers.append(tf.keras.layers.Conv2D(filters=1*depth, kernel_size=4, strides=2, activation=activation))
        self.conv_layers.append(tf.keras.layers.Conv2D(filters=2*depth, kernel_size=4, strides=2, activation=activation))
        self.conv_layers.append(tf.keras.layers.Conv2D(filters=4*depth, kernel_size=4, strides=2, activation=activation))
        self.conv_layers.append(tf.keras.layers.Conv2D(filters=8*depth, kernel_size=4, strides=2, activation=activation))

    @tf.function
    def call(self, input):
        input = tf.divide(tf.cast(input, tf.float32), tf.constant(255.)) - 0.5

        #z = self.input_layer(input)
        z = input

        for conv in self.conv_layers:
            z = conv(z)

        shape = tf.concat([tf.shape(input)[:-3], [32 * self.depth]], 0)

        return tf.reshape(z, shape)



class DreamerConvDecoder(tf.keras.Model):
    def __init__(self, output_dim=(3, 64, 64), depth=32, activation='relu'):
        super(DreamerConvDecoder, self).__init__()
        self.output_dim = output_dim
        self.depth = depth

        self.fc = tf.keras.layers.Dense(32*depth)

        self.conv_layers = []
        self.conv_layers.append(tf.keras.layers.Conv2DTranspose(filters=4*depth, kernel_size=5, strides=2, activation=activation))
        self.conv_layers.append(tf.keras.layers.Conv2DTranspose(filters=2*depth, kernel_size=5, strides=2, activation=activation))
        self.conv_layers.append(tf.keras.layers.Conv2DTranspose(filters=1*depth, kernel_size=6, strides=2, activation=activation))
        self.conv_layers.append(tf.keras.layers.Conv2DTranspose(filters=output_dim[0], kernel_size=6, strides=2))

    @tf.function
    def call(self, input):
        z = self.fc(input)
        z = tf.reshape(z, [-1, 1, 1, 32 * self.depth])
        for conv in self.conv_layers:
            z = conv(z)
        mean = tf.reshape(z, tf.concat([tf.shape(input)[:-1], self.output_dim], 0))

        return tfp.distributions.Independent(tfp.distributions.Normal(mean, 1), len(self.output_dim))

'''
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)#num_units: 400
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
'''

class DreamerDenseDecoder(tf.keras.Model):
    def __init__(self, hidden_units = (400, 400), distribution='normal', activation='elu'):
        super(DreamerDenseDecoder, self).__init__()
        self.distribtuion = distribution
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units[i], activation=activation))

        self.output_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, input):
        z = input
        for layer in self.hidden_layers:
            z = layer(z)

        z = self.output_layer(z)
        z = tf.reshape(z, tf.shape(input)[:-1])

        if self.distribution == 'normal':
            return tfp.distributions.Independent(tfp.distributions.Normal(z, 1), 1)
        elif self.distribution == 'binary':
            return tfp.distributions.Independent(tfp.distributions.Bernoulli(z), 1)
        else:
            raise NotImplementedError(self.distribution)



class DreamerActionDecoder(tf.keras.Model):
    def __init__(self, action_dim, hidden_units=(400, 400, 400, 400), distribution='tanh_normal', init_std=5.0, mean_scale=5, activation='elu'):
        super(DreamerActionDecoder, self).__init__()
        self.raw_init_std = np.log(np.exp(init_std) - 1)
        self.mean_scale = mean_scale
        self.distribution = distribution
        self.action_dim = action_dim

        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units[i], activation=activation))

        self.output_layer = tf.keras.layers.Dense(2 * action_dim)

    @tf.function
    def call(self, input):
        z = input
        for layer in self.hidden_layers:
            z = layer(z)

        if self.distribution == 'tanh_normal':
            output = self.output_layer(z)

            mean, std = output[:, :self.action_dim], output[:, self.action_dim:]
            mean = self.mean_scale * tf.tanh(mean / self.mean_scale)
            std = tf.nn.softplus(std + self.raw_init_std) + 1e-4

            dist = tfp.distributions.Normal(mean, std)
            dist = tfp.distributions.TransformedDistribution(dist, TanhBijector())
            dist = tfp.distributions.Independent(dist, 1)
            dist = SampleDist(dist)

        elif self.distribution == 'onehot':
            self.output_layer = tf.keras.layers.Dense(self.action_dim)
            output = self.output_layer(z)
            dist = OneHotDist(output)

        else:
            raise NotImplementedError(self.distribution)

        return dist


class RSSM(tf.keras.Model):
    def _init__(self, stoch=30, deter=200, hidden=200, act='elu'):
        super(RSSM, self).__init__()

        self.cell = tf.keras.layers.GRUCell(deter)
