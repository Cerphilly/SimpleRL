import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, state_dim,  feature_dim = 50, data_format='channels_first', activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.feature_dim = feature_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim), name='Input')

        self.conv_layers = []
        self.conv_layers.append(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation=activation, data_format=data_format))
        self.conv_layers.append(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation=activation, data_format=data_format))
        self.conv_layers.append(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation=activation, data_format=data_format))
        self.conv_layers.append(tf.keras.layers.Flatten())
        self.conv_layers.append(tf.keras.layers.Dense(feature_dim))
        self.conv_layers.append(tf.keras.layers.LayerNormalization())

    @tf.function
    def call(self, input):

        input = tf.divide(tf.cast(input, tf.float32),
                             tf.constant(255.))

        z = self.input_layer(input)

        for conv in self.conv_layers:
            z = conv(z)


        return z