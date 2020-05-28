#Generative Adversarial Imitation Learning, Ho and Ermon, 2016.
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


from Common.Buffer import Buffer
from Networks.Discriminator import Discriminator

class GAIL:
    def __init__(self, algorithm, state_dim, action_dim, max_action, min_action, save, load, discriminator=None, batch_size=100, buffer_size=1e6):
        pass
