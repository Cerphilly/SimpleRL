import tensorflow as tf

class Policy_network(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Policy_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim, ), name='Input')
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer, name='Layer{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(self.action_dim, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Output')

    @tf.function
    def call(self, input, use_tanh=True):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)

        if use_tanh == True:
            output = tf.nn.tanh(output)

        return output


class Q_network(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Q_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim + self.action_dim,), name='Input')
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, name='Layer{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Output')

    @tf.function
    def call(self, input1, input2):
        input = tf.concat([input1, input2], axis=1)

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)

        return output


class V_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units=(256, 256), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(V_network, self).__init__()
        self.state_dim = state_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim, ), name='Input')
        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, name='Layer{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name='Output')

    @tf.function
    def call(self, input):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)

        return output

