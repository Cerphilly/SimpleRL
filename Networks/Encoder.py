import tensorflow as tf

from Common.Utils import copy_weight

class PixelEncoder(tf.keras.Model):
    def __init__(self, state_dim,  feature_dim = 50, layer_num=2, filter_num=32, data_format='channels_first', activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(PixelEncoder, self).__init__()

        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.filter_num = filter_num

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim), name='Input')

        self.conv_layers = [tf.keras.layers.Conv2D(filter_num, kernel_size=3, strides=2, activation=activation, data_format=data_format)]
        for i in range(layer_num - 1):
            self.conv_layers.append(tf.keras.layers.Conv2D(filter_num, kernel_size=3, strides=1, activation=activation, data_format=data_format))

        self.conv_layers.append(tf.keras.layers.Flatten())
        self.fc = tf.keras.layers.Dense(feature_dim, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self.ln = tf.keras.layers.LayerNormalization()

    @tf.function
    def call(self, input, detach=False, activation=None):

        input = tf.divide(tf.cast(input, tf.float32),
                             tf.constant(255.))

        z = self.input_layer(input)

        for conv in self.conv_layers:
            z = conv(z)

        if detach == True:
            z = tf.stop_gradient(z)

        z = self.fc(z)
        z = self.ln(z)

        if activation == 'tanh':
            z = tf.nn.tanh(z)

        return z

    def copy_conv_weights(self, target):#copy this network's conv layers to target network's conv layers
                                        #to copy all layers, use utils's copy_weight(network, target_network)
        assert len(target.conv_layers) == len(self.conv_layers)

        for i in range(self.layer_num):
            copy_weight(self.conv_layers, target.conv_layers)



if __name__ == '__main__':
    a = PixelEncoder((3, 84, 84))
    b = PixelEncoder((3, 84, 84))
    input = tf.random.normal((1,3,84,84))
    b.copy_conv_weights(a)
    print(b(input), a(input))
    copy_weight(b, a)
    print(b(input), a(input))
