import tensorflow as tf
import gym
import pybullet_envs

import sys
import time
sys.path.append('../')

from Algorithms.REINFORCE import REINFORCE
from Algorithms.VPG import VPG
from Algorithms.PPO import PPO
from Algorithms.DDPG import DDPG
from Algorithms.TD3 import TD3
from Algorithms.TRPO import TRPO
from Algorithms.SAC_v1 import SAC_v1
from Algorithms.SAC_v2 import SAC_v2


class Online_Gym_trainer:
    def __init__(self, env, algorithm, max_action, min_action, render=True, max_episode = 1e6):

        self.env = env
        self.algorithm = algorithm

        self.max_action = max_action
        self.min_action = min_action

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

                if self.total_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()
                   next_observation, reward, done, _ = self.env.step(action)

                else:
                    action = self.algorithm.get_action(observation)
                    next_observation, reward, done, _ = self.env.step(self.max_action * action)

                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation


                if self.total_step >= self.algorithm.training_start:
                    self.algorithm.train(training_num=self.algorithm.training_step)

            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))


class Offline_Gym_trainer:
    def __init__(self, env, algorithm, max_action, min_action, render=True, max_episode=1e6):

        self.env = env
        self.algorithm = algorithm

        self.max_action = max_action
        self.min_action = min_action

        self.render = render

        self.max_episode = max_episode

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.random = False
        self.train_count = 0

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
                    self.env.render()

                if self.total_step <= self.algorithm.training_start:
                    action = self.env.action_space.sample()
                    next_observation, reward, done, _ = self.env.step(action)

                else:
                    action = self.algorithm.get_action(observation)
                    next_observation, reward, done, _ = self.env.step(self.max_action * action)

                self.episode_reward += reward
                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

            if self.total_step >= self.algorithm.training_start:
                self.algorithm.train(training_num=self.algorithm.training_step)

            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward,
                                                                                     self.local_step, self.total_step))


def main(cpu_only = False, force_gpu = True):
    if cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    if force_gpu == True:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    #env = gym.make("Pendulum-v0")
    #env = gym.make("MountainCarContinuous-v0")

    #env = gym.make("InvertedTriplePendulumSwing-v2")
    #env = gym.make("InvertedTriplePendulum-v2")
    #env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    #env = gym.make("InvertedPendulumSwing-v2")#around 10000 steps

    env = gym.make("InvertedPendulum-v2")

    #env = gym.make("Ant-v2")
    #env = gym.make("HalfCheetah-v2")
    #env = gym.make("Hopper-v2")
    #env = gym.make("Humanoid-v2")
    #env = gym.make("HumanoidStandup-v2")
    #env = gym.make("Reacher-v2")
    #env = gym.make("Swimmer-v2")
    #env = gym.make("Walker2d-v2")

    #env = gym.make("InvertedPendulumSwingupBulletEnv-v0")

    #env = gym.make("InvertedDoublePendulumBulletEnv-v0")
    #env = gym.make("InvertedDoublePendulumSwingupBulletEnv-v0")
    #env = gym.make("MinitaurBulletEnv-v0", render=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("Training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)

    #reinforce = REINFORCE(state_dim, action_dim, discrete=False)
    #vpg = VPG(state_dim, action_dim, discrete=False)
    trpo = TRPO(state_dim, action_dim, discrete=False)
    #ppo = PPO(state_dim, action_dim, discrete=False, mode='clip', clip=0.2)
    #ppo = PPO(state_dim, action_dim, discrete=False, mode='Adaptive KL', dtarg=0.01)
    #ppo = PPO(state_dim, action_dim, discrete=False, mode='Fixed KL', beta=3)

    #ddpg = DDPG(state_dim, action_dim)
    #td3 = TD3(state_dim, action_dim)

    #sac_v1 = SAC_v1(state_dim, action_dim)
    #sac_v2 = SAC_v2(state_dim, action_dim, auto_alpha=True)

    trainer = Offline_Gym_trainer(env=env, algorithm=trpo, max_action=max_action, min_action=min_action, render=True)
    trainer.run()



if __name__ == '__main__':
    main()