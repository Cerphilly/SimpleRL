import cv2
import numpy as np
import time

from Common.Utils import discrete_env, render_env
from Common.Logger import Logger

class Basic_trainer:
    def __init__(self, env, test_env, algorithm, max_action, min_action, args):
        self.domain_type = args.domain_type
        self.env_name = args.env_name

        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm

        self.logger = Logger(env, test_env, algorithm, max_action, min_action, args)

        self.max_action = max_action
        self.min_action = min_action

        self.discrete = discrete_env(env)
        self.render = args.render
        self.max_step = args.max_step

        self.eval = args.eval
        self.eval_episode = args.eval_episode
        self.eval_step = args.eval_step

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0

        self.train_mode = None

        if args.train_mode == 'offline':
            self.train_mode = self.offline_train

        elif args.train_mode == 'online':
            self.train_mode = self.online_train

            if not self.algorithm.training_step == 1:
                import warnings
                warnings.warn("online training means only 1 training step per each step")
                self.algorithm.training_step = 1

        assert self.train_mode is not None


    def offline_train(self, d, local_step):
        if d:
            return True
        return False

    def online_train(self, d, local_step):
        return True

    def evaluate(self):
        self.eval_num += 1
        episode = 0
        reward_list = []

        while True:
            if episode >= self.eval_episode:
                break
            episode += 1
            eval_reward = 0
            local_step = 0
            observation = self.test_env.reset()

            if '-ram-' in self.env_name:  # Atari Ram state
                observation = observation / 255.

            done = False

            while not done:
                local_step += 1

                if self.render == True:
                    render_env(self.env, self.env_name, self.domain_type, self.algorithm.name)

                action = self.algorithm.eval_action(observation)

                if self.discrete == False:
                    env_action = self.max_action * np.clip(action, -1, 1)
                else:
                    env_action = action

                next_observation, reward, done, _ = self.test_env.step(env_action)

                eval_reward += reward
                observation = next_observation

            reward_list.append(eval_reward)

            #self.logger.log_values([['Reward/Eval', eval_reward], ['Step/Eval', local_step]], mode='eval')
        '''
        print("Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f} ".format(
            sum(reward_list) / len(reward_list), max(reward_list), min(reward_list), np.std(reward_list)))
        '''
        self.logger.log_values([['Episode/Eval', self.eval_num], ['Reward/Eval', sum(reward_list) / len(reward_list)], ['Max_Reward/Eval', max(reward_list)], ['Min_Reward/Eval', min(reward_list)], ['Stddev_Reward/Eval', np.std(reward_list)]], mode='eval'),
        self.logger.results('eval')

    def run(self):
        while True:
            if self.total_step > self.max_step:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            observation = self.env.reset()
            done = False
            time_list = []

            while not done:
                self.local_step += 1
                self.total_step += 1
                start_time = time.time()

                if self.render == True:
                    render_env(self.env, self.env_name, self.domain_type, self.algorithm.name)

                if '-ram-' in self.env_name:  # Atari Ram state
                    observation = observation / 255.

                if self.total_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()
                   next_observation, reward, done, _ = self.env.step(action)

                else:
                    if self.algorithm.buffer.on_policy == False:
                        action = self.algorithm.get_action(observation)
                    else:
                        action, log_prob = self.algorithm.get_action(observation)

                    if self.discrete == False:
                        env_action = self.max_action * np.clip(action, -1, 1)
                    else:
                        env_action = action

                    next_observation, reward, done, _ = self.env.step(env_action)

                if self.local_step + 1 == 1000:
                    real_done = 0.
                else:
                    real_done = float(done)

                self.episode_reward += reward

                if self.env_name == 'Pendulum-v0':
                    reward = (reward + 8.1) / 8.1

                if self.algorithm.buffer.on_policy == False:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done)
                else:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done, log_prob)

                observation = next_observation

                if self.total_step >= self.algorithm.training_start and self.train_mode(done, self.local_step):
                    loss_list = self.algorithm.train(self.algorithm.training_step)
                    self.logger.log_values(loss_list, mode='train')

                if self.eval == True and self.total_step % self.eval_step == 0:
                    self.evaluate()

            self.logger.log_values([["Episode/Train", self.episode], ["Reward/Train", self.episode_reward], ["Step/Train", self.local_step], ["Total_Step/Train", self.total_step]], mode='train')
            #print("Train | Episode: {}, Reward: {:.2f} Local_step: {:<10} Total_step: {:<10}".format(self.episode, self.episode_reward, self.local_step, self.total_step))
            #print(self.logger.return_results('train'))
            #print(self.logger.return_results('loss'))
            #print("Train |", self.logger.return_results('train'))
            #print(self.logger.console_results('train'))
            self.logger.results('train')
