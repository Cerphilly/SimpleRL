import tensorflow as tf
import numpy as np
from Common.Utils import copy_weight

class PixelEncoder(tf.keras.Model):
    def __init__(self, obs_dim, feature_dim = 50, layer_num=2, filter_num=32, data_format='channels_first', activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(PixelEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.filter_num = filter_num

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.obs_dim), name='Input')#state_dim: (3,84,84)

        self.conv_layers = [tf.keras.layers.Conv2D(filter_num, kernel_size=3, strides=2, activation=activation, data_format=data_format, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)]
        for i in range(layer_num - 1):
            self.conv_layers.append(tf.keras.layers.Conv2D(filter_num, kernel_size=3, strides=1, activation=activation, data_format=data_format, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        self.conv_layers.append(tf.keras.layers.Flatten())
        self.fc = tf.keras.layers.Dense(feature_dim, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.ln = tf.keras.layers.LayerNormalization()

        self(tf.constant(np.zeros(shape = (1, *self.obs_dim), dtype=np.float32)))


    @tf.function
    def call(self, input, activation=None):

        input = tf.divide(tf.cast(input, tf.float32), tf.constant(255.))

        z = self.input_layer(input)

        for conv in self.conv_layers:
            z = conv(z)

        z = self.fc(z)
        z = self.ln(z)

        if activation == 'tanh':
            z = tf.nn.tanh(z)

        return z




if __name__ == '__main__':
    a = PixelEncoder((9, 84, 84), layer_num=4)
    b = PixelEncoder((3, 84, 84))
    a.save_weights()
    input = tf.random.normal((1,9,84,84))
    print(a(input))
    a.summary()
    print(a.conv_layers[0].shape)
