import tensorflow as tf
import numpy as np
import os
import datetime

class TFSaver:
    def __init__(self, algorithm_name, path='saved_models', save_name='test'):
        #algorithm_name = 'DQN', 'DDPG', etc
        #path: Algorithm_name_save_name_TF_datetime
        #log_path: Algorithm_name_savename_TF_datetime
        #put tensorboard thing in logs, put model weights in saved_models
        #weight save path: saved_models/algorithm_name_(save_name)_TF_datetime/actor, critic...

        self.summary_writer = tf.summary.create_file_writer(path)
        self.path = path
        self.save_name = save_name
        self.done = False

    def make_path(self):
        if os.path.isdir(os.path.join(self.path, self.save_name)):
            os.mkdir(os.path.join(self.path, self.save_name))

    def log(self, step, **kwargs):
        for key, value in kwargs.items():
            with self.summary_writer.as_default():
                tf.summary.scalar(key, value, step=step)

    def save_graph(self):
        if not self.done:
            with self.summary_writer.as_default():
                tf.summary.trace_export(name="train", step=0, profiler_outdir='summaries')
            self.done = True


    def save_weights(self, **kwargs):
        #network_list: [{'Actor': self.actor}, {'Critic: self.critic}, etc]
        for key, value in kwargs.items():
            value.save_weights(os.path.join(os.path.join(self.path, self.save_name), value), overwrite=False, save_format='h5')

    def save_buffer(self, buffer):
        np.savez_compressed('path+buffer', s = buffer.s, a = buffer.a, r = buffer.r, d = buffer.d, ns = buffer.ns)


    def load_weights(self, **kwargs):
        for key, value in kwargs.items():
            value.load_weights('path+key')
            print("{} loaded".format(key))

    def load_buffer(self):
        pass

