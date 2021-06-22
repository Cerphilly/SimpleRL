import dmc2gym
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import cv2

from Common.Utils import FrameStack
from Common.Logger import Logger


class Image_trainer:
    def __init__(self, env, test_env, algorithm, max_action, min_action, args):
        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm
        self.domain_type = args.domain_type

        self.max_action = max_action
        self.min_action = min_action

        self.render = args.render
        self.max_episode = args.max_episode

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

    def evaluate(self):
        self.eval_num += 1
        episode = 0
        reward_list = []

        while True:
            if episode >= self.eval_episode:
                break

            episode += 1
            eval_reward = 0
            observation = self.test_env.reset()
            done = False

            while not done:
                if self.render == True:
                    if self.domain_type == 'gym':
                        self.test_env.render()
                    else:
                        cv2.imshow("{}_{}".format(self.algorithm.name, self.test_env.unwrapped.spec.id), self.test_env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                action = self.algorithm.eval_action(observation)
                next_observation, reward, done, _ = self.test_env.step(self.max_action * action)

                eval_reward += reward
                observation = next_observation

            reward_list.append(eval_reward)

        print("Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}  ".format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list)))

        if self.log == True:
            self.writer.log('Reward/Test', sum(reward_list)/len(reward_list), self.eval_num)
            self.writer.log('Max Reward/Test', max(reward_list), self.eval_num)
            self.writer.log('Min Reward/Test', min(reward_list), self.eval_num)

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

                if self.eval == True and self.total_step % self.eval_step == 0:
                    self.evaluate()

                if self.render == True:
                    if self.domain_type == 'gym':
                        self.env.render()
                    else:
                        cv2.imshow("{}_{}".format(self.algorithm.name, self.env.unwrapped.spec.id), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                if self.total_step <= self.algorithm.training_start:
                   action = self.env.action_space.sample()
                   next_observation, reward, done, _ = self.env.step(action)

                else:
                    action = self.algorithm.get_action(observation)
                    next_observation, reward, done, _ = self.env.step(self.max_action * action)

                if self.local_step + 1 == self.env._max_episode_steps:
                    real_done = 0.
                else:
                    real_done = float(done)

                self.episode_reward += reward

                self.algorithm.buffer.add(observation, action, reward, next_observation, real_done)
                observation = next_observation

                if self.total_step >= self.algorithm.training_start and self.train_mode(done, self.local_step):
                    loss_list = self.algorithm.train(self.algorithm.training_step)
                    if self.log == True:
                        for loss in loss_list:
                            self.writer.log(loss[0], loss[1], self.total_step, str(self.episode))


            print("Train | Episode: {}, Reward: {:.2f}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))

            if self.log == True:
                self.writer.log('Reward/Train', self.episode_reward, self.episode)
                self.writer.log('Step/Train', self.local_step, self.episode)


# def main(cpu_only = False, force_gpu = True):
#     #device setting
#     #################################################################################
#     if cpu_only == True:
#         cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
#         tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')
#
#     if force_gpu == True:
#         gpu = tf.config.experimental.list_physical_devices('GPU')
#         tf.config.experimental.set_memory_growth(gpu[0], True)
#
#     FRAME_STACK = 3
#     IMAGE_SIZE = 84
#     PRE_IMAGE_SIZE = 100
#
#     env = dmc2gym.make(domain_name="cartpole", task_name='swingup', seed=np.random.randint(1, 9999), visualize_reward=False, from_pixels=True,
#                        height=PRE_IMAGE_SIZE, width=PRE_IMAGE_SIZE, frame_skip=8)#Pre image size for curl, image size for dbc
#     env = FrameStack(env, k=FRAME_STACK)
#
#     obs_shape = (3 * FRAME_STACK, IMAGE_SIZE, IMAGE_SIZE)
#     pre_aug_obs_shape = (3 * FRAME_STACK, PRE_IMAGE_SIZE, PRE_IMAGE_SIZE)
#     action_shape = env.action_space.shape[0]
#     max_action = env.action_space.high[0]
#     min_action = env.action_space.low[0]
#
#     # #algorithm = CURL_SACv1(obs_shape, action_shape)#frame_skip: 8, image_size: 100
#     # algorithm = CURL_SACv2(obs_shape, action_shape)#frame_skip: 8, image_size: 100
#     # #algorithm = CURL_TD3(obs_shape, action_shape)#frame_skip: 8, image_size: 100
#     #
#     #
#     # trainer = Image_trainer(env=env, algorithm=algorithm, max_action=max_action, min_action=min_action, train_mode='online', render=False)
#     # trainer.run()
#
#
# if __name__ == '__main__':
#     main()



