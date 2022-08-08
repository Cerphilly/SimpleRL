import cv2
import numpy as np

from Common.Utils import discrete_env, render_env, trim_float

class Basic_trainer:
    def __init__(self, env, test_env, algorithm, max_action, min_action, args):
        self.domain_type = args.domain_type
        self.env_name = args.env_name

        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm

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

        self.total_loss = {'Loss':{}}

        if args.train_mode == 'offline':
            self.train_mode = self.offline_train
        elif args.train_mode == 'online':
            self.train_mode = self.online_train
        elif args.train_mode == 'batch':
            self.train_mode = self.batch_train

        assert self.train_mode is not None

    def offline_train(self, d, local_step):
        if d:
            return True
        return False

    def online_train(self, d, local_step):
        return True

    def batch_train(self, d, local_step):  # VPG, TRPO, PPO only
        if d or local_step == self.algorithm.batch_size:
            return True

        return False

    def logger(self, loss_list):
        if loss_list is None:
            return



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

            if '-ram-' in self.env_name:  # Atari Ram state
                observation = observation / 255.

            done = False

            while not done:
                if self.render:
                    render_env(self.env, self.env_name, self.domain_type, self.algorithm.name)

                action = self.algorithm.eval_action(observation)

                if not self.discrete:
                    env_action = self.max_action * np.clip(action, -1, 1)
                else:
                    env_action = action

                next_observation, reward, done, _ = self.test_env.step(env_action)

                eval_reward += reward
                observation = next_observation

            reward_list.append(eval_reward)

        print("Eval  | Average Reward {:.2f} | Max reward: {:.2f} | Min reward: {:.2f} | Stddev reward: {:.2f} ".format(
            sum(reward_list) / len(reward_list), max(reward_list), min(reward_list), np.std(reward_list)))

    def run(self):
        loss_list = None

        while True:
            if self.total_step > self.max_step:
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

                if self.render:
                    render_env(self.env, self.env_name, self.domain_type, self.algorithm.name)

                if '-ram-' in self.env_name:  # Atari Ram state
                    observation = observation / 255.

                if self.total_step <= self.algorithm.training_start:
                    action = self.env.action_space.sample()
                    next_observation, reward, done, _ = self.env.step(action)

                else:
                    if not self.algorithm.buffer.on_policy:
                        action = self.algorithm.get_action(observation)
                    else:
                        action, log_prob = self.algorithm.get_action(observation)

                    if not self.discrete:
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

                if not self.algorithm.buffer.on_policy:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done)
                else:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done, log_prob)

                observation = next_observation

                if self.total_step >= self.algorithm.training_start and self.train_mode(done, self.local_step):
                    loss_list = self.algorithm.train(self.algorithm.training_step)
                    trim_float(loss_list)

                if self.eval and self.total_step % self.eval_step == 0:
                    self.evaluate()

            print("Train | Episode: {} | Reward: {:.2f} | Local_step: {} | Total_step: {} |".format(self.episode,
                                                                                                    self.episode_reward,
                                                                                                    self.local_step,
                                                                                                    self.total_step), loss_list)
