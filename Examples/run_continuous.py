import tensorflow as tf
import gym

import sys
sys.path.append('../')

from Algorithms.REINFORCE import REINFORCE
from Algorithms.VPG import VPG
from Algorithms.DDPG import DDPG
from Algorithms.TD3 import TD3
from Algorithms.SAC_v1 import SAC_v1
from Algorithms.SAC_v2 import SAC_v2

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

                if self.local_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()


                next_observation, reward, done, _ = self.env.step(action)

                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation

                if self.total_step >= self.algorithm.training_start:
                    self.algorithm.train(training_num=1)

            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))


class Offline_Gym_trainer:
    def __init__(self, env, algorithm, render, save, load, log, save_period, max_episode=1e6):

        self.env = env
        self.algorithm = algorithm
        #self.saver = Saver('SAC_v1', 'low_damping', log)

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

        #if self.load == True:
            #self.saver.load_weights(**self.algorithm.network_list)
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
                self.algorithm.train(training_num=self.local_step)


            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward,
                                                                                     self.local_step, self.total_step))

            if self.log == True:
                self.saver.log(self.episode, **{"reward": self.episode_reward, "local_step": self.local_step})

            if self.save == True and self.episode % self.save_period == 0:
                self.saver.save_weights(**self.algorithm.network_list)


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
    #env = gym.make("InvertedPendulumSwing-v2")#around 10000 steps.
    env = gym.make("InvertedPendulum-v2")

    #env = gym.make("Ant-v2")
    #env = gym.make("HalfCheetah-v2")
    #env = gym.make("Hopper-v2")
    #env = gym.make("Humanoid-v2")
    #env = gym.make("HumanoidStandup-v2")
    #env = gym.make("Reacher-v2")
    #env = gym.make("Swimmer-v2")
    #env = gym.make("Walker2d-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    print("Training of", env.unwrapped.spec.id)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)

    reinforce = REINFORCE(state_dim, action_dim, max_action, min_action, discrete=False)
    vpg = VPG(state_dim, action_dim, max_action, min_action, discrete=False)
    ddpg = DDPG(state_dim, action_dim, max_action, min_action)
    #td3 = TD3(state_dim, action_dim, max_action, min_action)
    #sac_v1 = SAC_v1(state_dim, action_dim, max_action, min_action)
    #sac_v2 = SAC_v2(state_dim, action_dim, max_action, min_action, auto_alpha=True)

    trainer = Offline_Gym_trainer(env=env, algorithm=vpg, render=True, save=False, load=False, log=False, save_period=100)
    trainer.run()



if __name__ == '__main__':
    main()