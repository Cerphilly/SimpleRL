import numpy as np
import gym

from Tensorflow2.Common.TFSaver import TFSaver

class Online_Gym_trainer:
    def __init__(self, env, algorithm, render=True, max_episode = 1e6):

        self.env = env
        self.algorithm = algorithm
        self.render = render
        self.max_episode = max_episode

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0


    def run(self):

        while True:
            if self.episode > self.max_episode:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0
            self.losses = 0

            observation = self.env.reset()
            done = False

            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.render == True:
                    self.env.render()

                action = self.algorithm.get_action(observation)

                if self.total_step <= self.algorithm.training_start:
                    action = self.env.action_space.sample()

                next_observation, reward, done, _ = self.env.step(action)

                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if self.total_step >= self.algorithm.training_start:
                    self.losses = self.algorithm.train(training_num=1)

            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}, Losses: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step, self.losses))


class Offline_Gym_trainer:
    def __init__(self, env, algorithm, render=True, save=False, load=False, log=False, save_period=10, max_episode=1e6):

        self.env = env
        self.algorithm = algorithm
        self.render = render
        self.save = save
        self.load = load
        self.log = log

        self.save_period = save_period
        self.max_episode = max_episode

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.random = False

    def run(self):

        if self.load == True:
            self.algorithm.saver.load_weights(**self.algorithm.network_list)


        while True:
            if self.episode > self.max_episode:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0
            self.losses = None

            observation = self.env.reset()
            done = False
            self.random = False

            while not done:
                self.local_step += 1
                self.total_step += 1


                if self.render == True:
                    self.env.render()

                action = self.algorithm.get_action(observation)

                if self.load == False and self.total_step <= self.algorithm.training_start:
                    action = self.env.action_space.sample()
                    self.random = True

                next_observation, reward, done, _ = self.env.step(action)
                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

            if self.total_step >= self.algorithm.training_start:
                self.losses = self.algorithm.train(training_num=self.local_step)


            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}, Losses: {}".format(self.episode, self.episode_reward,
                                                                                     self.local_step, self.total_step, self.losses))

            if self.log == True and self.random == False:
                self.algorithm.saver.log(self.episode, **{"reward": self.episode_reward[0], "actor_loss": self.losses[0], "critic1_loss": self.losses[1], "critic2_loss": self.losses[2], "v_loss": self.losses[3]})

            if self.save == True and self.episode % self.save_period == 0:
                self.algorithm.saver.save_weights(**self.algorithm.network_list)









