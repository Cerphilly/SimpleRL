'''
import tensorflow as tf
import gym
from gym.spaces import Discrete, Box
import pybullet_envs
import dmc2gym
import numpy as np

import sys
import datetime

from Algorithms.REINFORCE import REINFORCE
from Algorithms.VPG import VPG
from Algorithms.TRPO import TRPO
from Algorithms.PPO import PPO

from Algorithms.DQN import DQN
from Algorithms.DDQN import DDQN
from Algorithms.Dueling_DQN import Dueling_DQN

from Algorithms.DDPG import DDPG
from Algorithms.TD3 import TD3
from Algorithms.SAC_v1 import SAC_v1
from Algorithms.SAC_v2 import SAC_v2

from Algorithms.D2RL import D2RL_TD3, D2RL_SAC_v1, D2RL_SAC_v2
'''

import cv2

from Common.Logger import Logger

class State_trainer:
    def __init__(self, env, algorithm, max_action, min_action, args):
        self.domain_type = args.domain_type
        self.env = env
        self.algorithm = algorithm

        self.max_action = max_action
        self.min_action = min_action

        self.render = args.render
        self.max_episode = args.max_episode

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0

        self.train_mode = None

        if args.train_mode == 'offline':
            self.train_mode = self.offline_train
        elif args.train_mode == 'online':
            self.train_mode = self.online_train
        elif args.train_mode == 'batch':
            self.train_mode = self.batch_train

        assert self.train_mode is not None

        self.log = args.log
        if self.log == True:
            self.writer = Logger(env, algorithm, args, console=args.console, tensorboard=args.tensorboard)

    def offline_train(self, d, local_step):
        if d:
            return True
        return False

    def online_train(self, d, local_step):
        return True

    def batch_train(self, d, local_step):#VPG, TRPO, PPO only
        if d or local_step == self.algorithm.batch_size:
            return True
        return False

    def run(self):
        while True:
            if self.episode > self.max_episode:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            observation = self.env.reset()
            done = False

            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.render == True:
                    if self.domain_type == 'gym':
                        self.env.render()
                    else:
                        cv2.imshow("env", self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                if self.total_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()
                   next_observation, reward, done, _ = self.env.step(action)

                else:
                    action = self.algorithm.get_action(observation)
                    next_observation, reward, done, _ = self.env.step(self.max_action * action)

                #done = 0. if self.local_step + 1 == self.env._max_episode_steps else float(done)
                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if self.total_step >= self.algorithm.training_start and self.train_mode(done, self.local_step):
                    loss_list = self.algorithm.train(training_num=self.algorithm.training_step)
                    for loss in loss_list:
                        self.writer.log(loss[0], loss[1], self.total_step, str(self.episode))


            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {},".format(self.episode, self.episode_reward, self.local_step, self.total_step))

            self.writer.log('Reward/Train', self.episode_reward, self.episode)
            self.writer.log('Step/Train', self.local_step, self.episode)




