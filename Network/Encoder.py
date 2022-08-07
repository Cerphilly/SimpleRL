import tensorflow as tf
import numpy as np
from Common.Utils import copy_weight

class PixelEncoder(tf.keras.Model):
    def __init__(self, obs_dim, feature_dim=50, layer_num=4, filter_num=36, kernel_size=3, strides=(2, 1, 1, 1),
                 data_format='channels_first', activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(PixelEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.filter_num = filter_num

        if isinstance(filter_num, int):
            self.filter_num = (filter_num,) * self.layer_num

        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * self.layer_num

        self.strides = strides
        if isinstance(strides, int):
            self.strides = (strides,) * self.layer_num

        assert len(self.filter_num) == layer_num
        assert len(self.kernel_size) == layer_num
        assert len(self.strides) == layer_num

        self.data_format = data_format
        self.activation = activation

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.obs_dim), name='input') #state_dim: (3,84,84)

        self.conv_layers = []
        for i in range(layer_num):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(filters=self.filter_num[i], kernel_size=self.kernel_size[i], strides=self.strides[i],
                                       padding='valid', data_format=data_format,
                                       activation=activation, dtype=tf.float32, name='conv{}'.format(i)))

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(feature_dim, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.ln = tf.keras.layers.LayerNormalization()

        self(tf.zeros(shape = (1, *self.obs_dim), dtype=tf.float32))


    @tf.function
    def call(self, input, activation=None):

        input = tf.divide(tf.cast(input, tf.float32), tf.constant(255.))

        z = self.input_layer(input)

        for conv in self.conv_layers:
            z = conv(z)

        z = self.flatten(z)
        z = self.fc(z)
        z = self.ln(z)

        if activation == 'tanh':
            z = tf.nn.tanh(z)

        return z




if __name__ == '__main__':
    a = PixelEncoder((3, 84, 84), layer_num=4)
    """
    (3, 3, 9, 32)
    (32,)
    (3, 3, 32, 32)
    (32,)
    (3, 3, 32, 32)
    (32,)
    (3, 3, 32, 32)
    (32,)
    (39200, 50)
    (50,)
    (50,)
    (50,)    
    """
    b = PixelEncoder((3, 84, 84))
    input = tf.random.normal((1,3,84,84))
    print(a(input))
    a.summary()

