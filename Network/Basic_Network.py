import tensorflow as tf

class Policy_network(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):

        super(Policy_network, self).__init__()

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


        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim,), dtype=tf.float32, name='Input')
        self.hidden_layers = []

        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(units=hidden_units[i], activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Dense{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(units=self.action_dim, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Output')

        self(tf.zeros(shape=(1,) + (self.state_dim,), dtype=tf.float32))


    @tf.function
    def call(self, input, activation=None):

        z = self.input_layer(input)

        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)
        #tf.keras.activations.get(None): linear activation
        #'elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh'
        output = tf.keras.activations.get(activation)(output)

        return output

class Q_network(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):

        super(Q_network, self).__init__()

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

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.action_dim,), dtype=tf.float32, name='Input')
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(units=hidden_units[i], activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Dense{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(units=1, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Output')

        self(tf.zeros(shape=(1,) + (self.state_dim,), dtype=tf.float32), tf.zeros(shape=(1,) + (self.action_dim,), dtype=tf.float32))

    @tf.function
    def call(self, input1, input2, activation=None):
        input = tf.concat([input1, input2], axis=-1)
        z = self.input_layer(input)

        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)
        # tf.keras.activations.get(None): linear activation
        # 'elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh'
        output = tf.keras.activations.get(activation)(output)

        return output


class V_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units=(256, 256), activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):

        super(V_network, self).__init__()
        self.state_dim = state_dim

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

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim, ), dtype=tf.float32, name='Input')
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(units=hidden_units[i], activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Dense{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(1, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      dtype=tf.float32, name='Output')


        self(tf.zeros(shape=(1,) + (self.state_dim,), dtype=tf.float32))

    @tf.function
    def call(self, input, activation=None):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)
        # tf.keras.activations.get(None): linear activation
        # 'elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh'
        output = tf.keras.activations.get(activation)(output)

        return output


if __name__ == '__main__':
    from tensorflow.python.keras import activations

    a = Q_network(5, 1)
    a.summary()
    exit()
    #print(tf.keras.activations.get(None)(-tf.ones((1, 10))))
    print(a(-tf.ones((1, 10))))
    print(a(-tf.ones((1, 10)), 'tanh'))
    a.summary()
    print(a.layers)
    #tf.keras.utils.plot_model(a, "1.png", show_shapes=True, show_dtype=True, expand_nested=True, show_layer_activations=True)
    #print(tf.keras.utils.serialize_keras_object(a))