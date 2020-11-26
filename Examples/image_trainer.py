import dmc2gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Algorithms.ImageRL.CURL import CURL_SACv1, CURL_SACv2, CURL_TD3
from Common.Utils import FrameStack
from Common.Buffer import Buffer

class Image_trainer:
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
        else:
            print("no training mode")

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
                    plt.imshow(np.transpose(observation, [1,2,0])[:,:,-3])
                    plt.show(block=False)
                    plt.pause(0.001)

                if self.total_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()
                   next_observation, reward, done, _ = self.env.step(action)

                else:
                    action = self.algorithm.get_action(observation)
                    next_observation, reward, done, _ = self.env.step(self.max_action * action)

                done = 0. if self.local_step + 1 == self.env._max_episode_steps else float(done)

                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, done)
                observation = next_observation


                if self.total_step >= self.algorithm.training_start and self.train_mode(done, self.local_step):
                    self.algorithm.train(self.local_step)


            print("Episode: {}, Reward: {}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))


def main(cpu_only = False, force_gpu = True):
    #device setting
    #################################################################################
    if cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    if force_gpu == True:
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    FRAME_STACK = 3
    IMAGE_SIZE = 84
    PRE_IMAGE_SIZE = 100

    env = dmc2gym.make(domain_name="cartpole", task_name="swingup", visualize_reward=False, from_pixels=True,
                       height=PRE_IMAGE_SIZE, width=PRE_IMAGE_SIZE, frame_skip=8)
    env = FrameStack(env, k=FRAME_STACK)

    obs_shape = (3 * FRAME_STACK, IMAGE_SIZE, IMAGE_SIZE)
    pre_aug_obs_shape = (3 * FRAME_STACK, PRE_IMAGE_SIZE, PRE_IMAGE_SIZE)
    action_shape = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    #algorithm = CURL_SACv1(obs_shape, action_shape)
    algorithm = CURL_SACv2(obs_shape, action_shape)
    #algorithm = CURL_TD3(obs_shape, action_shape)

    trainer = Image_trainer(env=env, algorithm=algorithm, max_action=max_action, min_action=min_action, train_mode='online', render=False)
    trainer.run()


if __name__ == '__main__':
    main()



