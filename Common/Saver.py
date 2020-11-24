import tensorflow as tf
import numpy as np
import os
import datetime

class Saver:
    def __init__(self, algorithm_name, save_name='test', log=False):
        #example: Algorithm_name_TF2_save_name
        self.path = 'SimpleRL/saved_models'

        self.save_name = save_name
        self.algorithm_name = algorithm_name

        self.make_path()
        if log == True:
            self.summary_writer = tf.summary.create_file_writer(os.path.join(self.path, '{}_{}'.format(self.algorithm_name, self.save_name)))


    def make_path(self):
        if not os.path.isdir(os.path.join(self.path, '{}_{}'.format(self.algorithm_name, self.save_name))):
            os.mkdir(os.path.join(self.path, '{}_{}'.format(self.algorithm_name, self.save_name)))

    def log(self, step, **kwargs):
        for key, value in kwargs.items():
            with self.summary_writer.as_default():
                tf.summary.scalar(key, value, step=step)

    def save_weights(self, **kwargs):
        #network_list: [{'Actor': self.actor}, {'Critic: self.critic}, etc]
        for key, value in kwargs.items():
            value.save_weights(os.path.join(os.path.join(self.path, '{}_{}'.format(self.algorithm_name, self.save_name)), key), overwrite=True, save_format='tf')

    def save_buffer(self, buffer):#unfinished
        np.savez_compressed('path+buffer', s = buffer.s, a = buffer.a, r = buffer.r, d = buffer.d, ns = buffer.ns)


    def load_weights(self, **kwargs):
        for key, value in kwargs.items():
            value.load_weights(os.path.join(os.path.join(self.path, '{}_{}'.format(self.algorithm_name, self.save_name)), key))
            print("{} loaded".format(key))

    def load_buffer(self):#unfinished
        pass

