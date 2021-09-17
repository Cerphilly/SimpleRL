import tensorflow as tf

class CURL(tf.keras.Model):
    def __init__(self, z_dim, batch_size):
        super(CURL, self).__init__()

        self.batch_size = batch_size
        self.W = tf.Variable(initial_value=tf.random.normal((z_dim, z_dim)), trainable=True)

    @tf.function
    def encode(self, x, encoder, target_encoder, detach=False, ema=False):#not used
        if ema == True:
            z_out = tf.stop_gradient(target_encoder(x))
        else:
            z_out = encoder(x)

        if detach == True:
            z_out = tf.stop_gradient(z_out)

        return z_out

    @tf.function
    def compute_logits(self, z_a, z_pos):
        Wz = tf.matmul(self.W, tf.transpose(z_pos))
        logits = tf.matmul(z_a, Wz)
        logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)

        return logits

