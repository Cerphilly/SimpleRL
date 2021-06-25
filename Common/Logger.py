import tensorflow as tf
import tensorboard
import numpy as np
import os, sys
import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Logger:
    def __init__(self, env, algorithm, args, dir_name=None, file=True, tensorboard=False, numpy=True, save_model=False, histogram=False):
        self.env = env
        self.algorithm = algorithm
        self.file = file
        self.tensorboard = tensorboard
        self.numpy = numpy
        self.save_model = save_model
        self.histogram = histogram

        if dir_name == None:
            self.current_time = self.log_time()
            self.dir_name = 'C:/Users/cocel/PycharmProjects/SimpleRL/Logs/' + env.unwrapped.spec.id + '-' +  algorithm.name + '-' +  self.current_time
            print("Log dir: ", self.dir_name)
            os.mkdir(self.dir_name)
            self.save_hyperparameter(args)

        else:
            self.dir_name = os.path.join('./Logs/', dir_name)

        if tensorboard:
            self.writer = tf.summary.create_file_writer(logdir=self.dir_name)

        if file:
            self.train_episode = 1
            self.test_episode = 1
            self.train_dict = {}
            self.test_dict = {}
            self.train_log = os.path.join(self.dir_name, 'train.log')
            self.test_log = os.path.join(self.dir_name, 'eval.log')

        if numpy:
            self.train_episode2 = 1#????
            self.test_episode2 = 1
            self.train_reward = [['Episode', 'Reward', 'Step', 'Total Step']]
            self.current_train = [self.train_episode2, 0, 0, 0]
            self.test_reward = [['Episode', 'Reward', 'Stddev Reward', 'Max Reward', 'Min Reward']]
            self.current_test = [self.test_episode2, 0, 0, 0, 0]
            self.train_numpy = os.path.join(self.dir_name, 'train_reward.npy')
            self.test_numpy = os.path.join(self.dir_name, 'test_reward.npy')

        if save_model:
            self.model_dir = os.path.join(self.dir_name, 'models')
            os.mkdir(self.model_dir)


    def set_type(self, tag, values):
        if tag.split('/')[0] == 'Reward':
            return values
        elif tag.split('/')[0] == 'Loss':
            return values
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
        if self.numpy:
            self.write_numpy(tag, values, step)
        if self.file:
            self.write_file(tag, values, step)
        if self.tensorboard:
            self.write_tensorboard(tag, values, step, description)

    def write_numpy(self, tag, values, global_step):
        if len(tag.split('/')) != 2:
            return

        if tag.split('/')[1] == 'Train':
            if self.train_episode2 == global_step:
                for i in range(len(self.train_reward[0])):
                    if tag.split('/')[0] == self.train_reward[0][i]:
                        self.current_train[i] = values
            else:
                self.train_reward = np.concatenate((np.array(self.train_reward), np.array([self.current_train])), axis=0)
                np.save(self.train_numpy, self.train_reward)

                self.train_episode2 = global_step
                self.current_train =  [self.train_episode2, 0, 0, 0]
                for i in range(len(self.train_reward[0])):
                    if tag.split('/')[0] == self.train_reward[0][i]:
                        self.current_train[i] = values

        elif tag.split('/')[1] == 'Test':
            if self.test_episode2 == global_step:
                for i in range(len(self.test_reward[0])):
                    if tag.split('/')[0] == self.test_reward[0][i]:
                        print(i)
                        self.current_test[i] = values
            else:
                self.test_reward = np.concatenate((np.array(self.test_reward), np.array([self.current_test])), axis=0)
                np.save(self.test_numpy, self.test_reward)

                self.test_episode2 = global_step
                self.current_test = [self.test_episode2, 0, 0, 0, 0]
                for i in range(len(self.test_reward[0])):
                    if tag.split('/')[0] == self.test_reward[0][i]:
                        self.current_test[i] = values

    def write_file(self, tag, values, global_step):
        #Train | Episode | Reward | Loss | Time
        if len(tag.split('/')) != 2:
            return

        if tag.split('/')[1] == 'Train':
            if self.train_episode == global_step:
                self.train_dict[tag.split('/')[0]] = values
            else:
                with open(self.train_log, 'a') as f:
                    f.write(str(self.train_dict) + '\n')
                self.train_episode = global_step
                self.train_dict = {}
                self.train_dict[tag.split('/')[0]] = values

        elif tag.split('/')[1] == 'Test':
            if self.test_episode == global_step:
                self.test_dict[tag.split('/')[0]] = values
            else:
                with open(self.test_log, 'a') as f:
                    f.write(str(self.test_dict) + '\n')
                self.test_episode = global_step
                self.test_dict = {}
                self.test_dict[tag.split('/')[0]] = values


    def write_tensorboard(self, tag, values, step, description):#global_step: episode
        #Reward/Train, Reward/Test, Loss/Train, Duration/Train
        with self.writer.as_default():
            values = self.set_type(tag, values)
            tf.summary.scalar(tag, values, step, description)






