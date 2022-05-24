import tensorflow as tf


class PixelEncoder(tf.keras.Model):
    def __init__(self, obs_dim, feature_dim = 50, layer_num=4, filter_num=36, kernel_size=3, strides=(2, 1, 1, 1), data_format='channels_last', activation='relu',
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        super(PixelEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.layer_num = layer_num
        self.filter_num = filter_num
        if isinstance(filter_num, int):
            self.filter_num = (filter_num, ) * self.layer_num

        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, ) * self.layer_num

        self.strides = strides
        if isinstance(strides, int):
            self.strides = (strides, ) * self.layer_num

        assert len(self.filter_num) == layer_num
        assert len(self.kernel_size) == layer_num
        assert len(self.strides) == layer_num

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

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.obs_dim), dtype=tf.float32, name='Input')#state_dim: (3,84,84)

        self.conv_layers = []
        for i in range(layer_num):
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=self.kernel_size[i], strides=self.strides[i], padding='valid',  data_format=data_format,
                                                           activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                                           kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, dtype=tf.float32, name='Conv{}'.format(i)))

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=feature_dim, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Output')

        self.ln = tf.keras.layers.LayerNormalization()

        self(tf.zeros(shape = (1, *self.obs_dim), dtype=tf.float32))

    @tf.function
    def call(self, input, activation=None):

        #input: [0 ~ 255] int
        input = tf.divide(tf.cast(input, tf.float32), tf.constant(255.))

        z = self.input_layer(input)

        for conv in self.conv_layers:
            z = conv(z)

        z = self.flatten(z)
        z = self.fc(z)
        z = self.ln(z)

        # tf.keras.activations.get(None): linear activation
        # 'elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh'
        output = tf.keras.activations.get(activation)(z)

        return output


if __name__ == '__main__':

    a = PixelEncoder((100, 100, 5))
    a.summary()
    for conv in a.conv_layers:
        print(conv.kernel.shape, conv.strides)
    #print(tf.keras.activations.get(None)(-tf.ones((1, 10))))
    import time
    c = time.time()
    print(a(-tf.ones((1, 100, 100, 5))).shape)

    #print(a(-tf.ones((1, 10)), 'relu'))
    a.summary()

