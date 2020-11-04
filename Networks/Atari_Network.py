import tensorflow as tf
import numpy as np

class Atari_Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Atari_Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim), name='Input')

        self.conv_layers = []
        self.conv_layers.append(tf.keras.layers.Conv2D(32, kernel_size=(8,8), strides=(4,4), padding='valid', activation='relu'))
        self.conv_layers.append(tf.keras.layers.Conv2D(32, kernel_size=(4,4), strides=(2,2), padding='valid', activation='relu'))
        #self.conv_layers.append(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        self.conv_layers.append(tf.keras.layers.Flatten())

        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer, name='Layer{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(self.action_dim, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Output')

    @tf.function
    def call(self, input, activation='tanh'):
        z = self.input_layer(input)
        for conv in self.conv_layers:
            z = conv(z)
        #print(z.shape[-1])
        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)

        if activation == 'tanh':
            output = tf.nn.tanh(output)

        elif activation == 'softmax':
            output = tf.nn.softmax(output)

        return output

class Atari_V_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units=(64, 64), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Atari_V_network, self).__init__()

        self.state_dim = state_dim

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.state_dim), name='Input')

        self.conv_layers = []
        self.conv_layers.append(tf.keras.layers.Conv2D(32, kernel_size=(8,8), strides=(4,4), padding='valid', activation='relu'))
        self.conv_layers.append(tf.keras.layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), padding='valid', activation='relu'))
        #self.conv_layers.append(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
        self.conv_layers.append(tf.keras.layers.Flatten())

        self.hidden_layers = []
        for i in range(len(hidden_units)):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units[i], activation=activation, kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer, name='Layer{}'.format(i)))

        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer,
                                                  bias_initializer=bias_initializer, name='Output')

    @tf.function
    def call(self, input):
        z = self.input_layer(input)
        for conv in self.conv_layers:
            z = conv(z)

        for layer in self.hidden_layers:
            z = layer(z)

        output = self.output_layer(z)

        return output

if __name__ == '__main__':
    a = Atari_Actor((500, 500, 3), 2)
    b = Atari_V_network((500, 500, 3))
    temp = np.random.rand(1, 500, 500, 3)
    print(temp)
    print(a(temp))
    print(b(temp))
    a.summary()
    b.summary()
    import gym
    env = gym.make("Pong-v4")
    print(env.reset().shape)