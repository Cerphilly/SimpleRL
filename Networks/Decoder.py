import tensorflow as tf
import numpy as np

OUT_DIM = {2: 39, 4: 35, 6: 31}

class PixelDecoder(tf.keras.Model):
    def __init__(self, obs_dim, feature_dim = 50, layer_num=2, filter_num=32, data_format='channels_first', activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(PixelDecoder, self).__init__()

        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.filter_num = filter_num

        self.out_dim = OUT_DIM[layer_num]

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.feature_dim), name='Input')#input shape: (b, feature_dim), output shape: (b, 3, 84, 84), etc

        self.fc = tf.keras.layers.Dense(self.filter_num * self.out_dim * self.out_dim, activation = activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.reshape = tf.keras.layers.Reshape(target_shape=(self.filter_num, self.out_dim, self.out_dim))

        self.deconv_layers = []
        for i in range(self.layer_num - 1):
            self.deconv_layers.append(tf.keras.layers.Conv2DTranspose(filters=self.filter_num, kernel_size=3, strides=1, data_format=data_format, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        self.deconv_layers.append(tf.keras.layers.Conv2DTranspose(filters=self.obs_dim[0], kernel_size=3, strides=2, output_padding=1, data_format=data_format, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

        self(tf.constant(np.zeros(shape=(1,) + (self.feature_dim,), dtype=np.float32)))

    @tf.function
    def call(self, feature):
        z = self.input_layer(feature)

        z = self.fc(z)
        z = self.reshape(z)

        for deconv in self.deconv_layers:
            z = deconv(z)

        return z

if __name__ == '__main__':
    b = PixelDecoder((9, 84, 84), layer_num=4)
    input = tf.random.normal((3,50))
    b(input)
    print(b(input).shape)
    b.summary()