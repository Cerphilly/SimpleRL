import tensorflow as tf

from Network.Basic_Network import Policy_network, Q_network, V_network

class D2RL_Policy(Policy_network):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256, 256, 256), activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):

        super(D2RL_Policy, self).__init__(state_dim=state_dim, action_dim=action_dim, hidden_units=hidden_units, activation=activation,
                                          use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)

    @tf.function
    def call(self, input, activation=None):

        z = self.input_layer(input)

        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)
        output = tf.keras.activations.get(activation)(output)

        return output


class D2RL_Q(Q_network):
    def __init__(self, state_dim, action_dim, hidden_units=(256, 256, 256, 256), activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        super(D2RL_Q, self).__init__(state_dim=state_dim, action_dim=action_dim, hidden_units=hidden_units, activation=activation,
                                          use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)

    @tf.function
    def call(self, input1, input2, activation=None):
        input = tf.concat([input1, input2], axis=1)

        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)
        output = tf.keras.activations.get(activation)(output)

        return output

class D2RL_V(V_network):
    def __init__(self, state_dim, hidden_units=(256, 256, 256, 256), activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        super(D2RL_V, self).__init__(state_dim=state_dim, hidden_units=hidden_units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                          bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)


    @tf.function
    def call(self, input, activation=None):
        z = self.input_layer(input)
        for layer in self.hidden_layers:
            z = layer(z)
            if layer != self.hidden_layers[-1]:
                z = tf.concat([z, tf.cast(input, tf.float32)], axis=1)

        output = self.output_layer(z)
        output = tf.keras.activations.get(activation)(output)

        return output