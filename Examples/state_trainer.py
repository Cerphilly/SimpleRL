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


class Gym_trainer:
    def __init__(self, env, algorithm, max_action, min_action,  train_mode, render=True, max_episode = 1e6):
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

        if train_mode == 'offline':
            self.train_mode = self.offline_train
        elif train_mode == 'online':
            self.train_mode = self.online_train
        elif train_mode == 'batch':
            self.train_mode = self.batch_train

        assert self.train_mode is not None

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
                    self.env.render()

                if self.total_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()
                   next_observation, reward, done, _ = self.env.step(action)

                else:
                    action = self.algorithm.get_action(observation)

                    next_observation, reward, done, _ = self.env.step(self.max_action * action)

                    #done: terminal state 1 True nonterminal state 0 False

                done = 0. if self.local_step + 1 == self.env._max_episode_steps else float(done)
                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation


                if self.total_step >= self.algorithm.training_start and self.train_mode(done, self.local_step):
                    self.algorithm.train(training_num=self.algorithm.training_step)

                # if self.total_step % 10000 == 1:
                #     file_path = 'C:/Users/cocel/PycharmProjects/SimpleRL/Saved_models/Humanoid/'
                #     layer_name = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
                #     for layer in range(len(self.algorithm.actor.trainable_variables)):
                #         np.save(file_path + layer_name[layer], self.algorithm.actor.trainable_variables[layer].numpy())



            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {},".format(self.episode, self.episode_reward, self.local_step, self.total_step))


def main(cpu_only = False, force_gpu = True):
    #device setting
    #################################################################################
    if cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    if force_gpu == True:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    #################################################################################

    #discrete env
    #################################################################################
    #env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")
    #env = gym.make("Acrobot-v1")

    #################################################################################
    #continuous env
    #################################################################################
    #env = gym.make("Pendulum-v0")
    #env = gym.make("MountainCarContinuous-v0")

    #env = gym.make("InvertedTriplePendulumSwing-v2")
    #env = gym.make("InvertedTriplePendulum-v2")
    #env = gym.make("InvertedDoublePendulumSwing-v2")
    #env = gym.make("InvertedDoublePendulum-v2")
    env = gym.make("InvertedPendulumSwing-v2")#around 10000 steps

    #env = gym.make("InvertedPendulum-v2")

    #env = gym.make("Ant-v2")
    #env = gym.make("HalfCheetah-v2")
    #env = gym.make("Hopper-v2")
    #env = gym.make("Humanoid-v3")
    #env = gym.make("HumanoidStandup-v2")
    #env = gym.make("Reacher-v2")
    #env = gym.make("Swimmer-v2")
    #env = gym.make("Walker2d-v2")

    #env = dmc2gym.make("cartpole", "swingup", seed=np.random.randint(1, 1000))

    #################################################################################

    #env setting
    #################################################################################
    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
        max_action = 1
        min_action = 1
        discrete = True
    elif isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        min_action = env.action_space.low[0]
        discrete = False
    else:
        raise NotImplementedError
    #################################################################################

    #algorithm for discrete env
    #################################################################################
    #algorithm = DQN(state_dim, action_dim)
    #algorithm = DDQN(state_dim, action_dim)
    #algorithm = Dueling_DQN(state_dim, action_dim)

    #################################################################################

    #algorithm for continuous env
    #################################################################################
    #algorithm = DDPG(state_dim, action_dim)
    #algorithm = TD3(state_dim, action_dim)
    #algorithm = SAC_v1(state_dim, action_dim)
    #algorithm = SAC_v2(state_dim, action_dim, alpha=0.05, train_alpha=False, training_start=10000)
    #algorithm = D2RL_SAC_v1(state_dim, action_dim)
    #algorithm = D2RL_SAC_v2(state_dim, action_dim, train_alpha=True)
    algorithm = D2RL_TD3(state_dim, action_dim, training_step=1)
    #d2rl_algorithm(algorithm)
    #print(algorithm.actor, algorithm.critic, algorithm.target_actor, algorithm.target_critic)
    #d2rl_algorithm(algorithm)
    #algorithm for both env
    #################################################################################
    #offline training only for REINFORCE, VPG, TRPO, PPO
    #algorithm = TRPO(state_dim, action_dim, discrete)
    #algorithm = PPO(state_dim, action_dim, discrete, mode='clip', clip=0.2)
    #algorithm= PPO(state_dim, action_dim, discrete, mode='Adaptive KL', dtarg=0.01)
    #algorithm = PPO(state_dim, action_dim, discrete, mode='Fixed KL', beta=3)
    #algorithm = REINFORCE(state_dim, action_dim, discrete)
    #algorithm = VPG(state_dim, action_dim, discrete)

    #################################################################################
    print("Training of", env.unwrapped.spec.id)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)
    print("Discrete: ", discrete)

    trainer = Gym_trainer(env=env, algorithm=algorithm, max_action=max_action, min_action=min_action, train_mode='online', render=False)
    trainer.run()



if __name__ == '__main__':
    main()


