import tensorflow as tf
import tensorboard
import numpy as np
import os, sys
import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Logger:
    def __init__(self, env, algorithm, args, dir_name=None, console=True, tensorboard=False):
        self.env = env
        self.algorithm = algorithm
        self.console = console
        self.tensorboard = tensorboard

        if tensorboard:
            if dir_name == None:
                self.current_time = self.log_time()
                self.dir_name = 'C:/Users/cocel/PycharmProjects/SimpleRL/Logs/' + env.unwrapped.spec.id + '-' +  algorithm.name + '-' +  self.current_time
                print("Log dir: ", self.dir_name)
                os.mkdir(self.dir_name)
                self.save_hyperparameter(args)

            else:
                self.dir_name = os.path.join('./Logs/', dir_name)

            self.writer = tf.summary.create_file_writer(logdir=self.dir_name)

        if console:
            self.current_step = 0

    def set_type(self, tag, values):
        if tag.split('/')[0] == 'Reward':
            return round(values, 2)
        elif tag.split('/')[0] == 'Loss':
            return round(values, 4)
        elif tag.split('/')[0] == 'Duration':
            return round(values, 2)
        else:
            return values

    def log_time(self):
        #YYYY-M-D-H-M-S
        current_time = datetime.datetime.now()
        output = str(current_time.year) + '-' + str(current_time.month) + '-' + str(current_time.day) + '-'
        output += str(current_time.hour) + '-' + str(current_time.minute) + '-' + str(current_time.second)

        return output

    def save_hyperparameter(self, args):
        with open((os.path.join(self.dir_name, 'hyperparameters.json')), 'w') as f:
            json.dump(vars(args), f, sort_keys=False, indent=5)

    def log(self, tag, values, step, description=None):
        if self.console:
            self.write_console(tag, values, step)
        if self.tensorboard:
            self.write_tensorboard(tag, values, step, description)

    def write_console(self, tag, values, global_step):
        #Train | Episode | Reward | Loss | Time

        pass

    def write_tensorboard(self, tag, values, step, description):#global_step: episode
        #Reward/Train, Reward/Test, Loss/Train, Duration/Train
        with self.writer.as_default():
            values = self.set_type(tag, values)
            tf.summary.scalar(tag, values, step, description)



