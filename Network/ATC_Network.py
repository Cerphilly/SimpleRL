import tensorflow as tf

#linear global compressor g_\phi to produce a small latent code vector c_t
class Compressor(tf.keras.Model):
    pass

#residual predictor MLP h_psi, which acts as an implicit forwrad model to advance the code p_t = h_\psi(c_t) + c_t
class Predictor(tf.keras.Model):
    pass