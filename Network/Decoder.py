import tensorflow as tf

OUT_DIM = {2: 39, 4: 35, 6: 31}

class PixelDecoder(tf.keras.Model):
    def __init__(self, obs_dim, feature_dim = 50, layer_num=4, filter_num=32, data_format='channels_first', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        super(PixelDecoder, self).__init__()

        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.filter_num = filter_num

        assert layer_num in OUT_DIM.keys(), "self.out_dim needs to be manually calculated"

        self.data_format = data_format
        self.activation = activation
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.out_dim = OUT_DIM[layer_num]

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.feature_dim), name='Input')

        self.fc = tf.keras.layers.Dense(units=self.filter_num * self.out_dim * self.out_dim, activation = activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                        dtype=tf.float32, name='Dense')
        self.reshape = tf.keras.layers.Reshape(target_shape=(self.filter_num, self.out_dim, self.out_dim) if self.data_format == 'channels_first' else (self.out_dim, self.out_dim, self.filter_num))

        self.deconv_layers = []
        for i in range(self.layer_num - 1):
            self.deconv_layers.append(tf.keras.layers.Conv2DTranspose(filters=self.filter_num, kernel_size=3, strides=1, data_format=data_format, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                        dtype=tf.float32, name='Conv2DTranspose{}'.format(i)))

        self.deconv_layers.append(tf.keras.layers.Conv2DTranspose(filters=self.obs_dim[0] if self.data_format == 'channels_first' else self.obs_dim[-1],
                                                                  kernel_size=3, strides=2, output_padding=1, data_format=data_format, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                        dtype=tf.float32, name='Conv2DTranspose{}'.format(self.layer_num - 1)))

        self(tf.zeros(shape=(1,) + (self.feature_dim,), dtype=tf.float32))

    @tf.function
    def call(self, feature):
        z = self.input_layer(feature)

        z = self.fc(z)
        z = self.reshape(z)

        for deconv in self.deconv_layers:
            z = deconv(z)

        return z

if __name__ == '__main__':
    b = PixelDecoder((84, 84, 9), layer_num=4, data_format='channels_last')
    b.summary()
    input = tf.random.normal((3,50))
    b(input)
    print(b(input).shape)
    b.summary()