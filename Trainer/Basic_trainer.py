import cv2
import numpy as np

from Common.Logger import Logger

class Basic_trainer:
    def __init__(self, env, test_env, algorithm, max_action, min_action, args):
        self.domain_type = args.domain_type
        self.env_name = args.env_name
        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm

        self.max_action = max_action
        self.min_action = min_action

        self.discrete = args.discrete
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
        elif args.train_mode == 'batch':
            self.train_mode = self.batch_train

        assert self.train_mode is not None

        self.log = args.log

        self.model = args.model
        self.model_freq = args.model_freq
        self.buffer = args.buffer
        self.buffer_freq = args.buffer_freq

        if self.log == True:
            self.logger = Logger(env, algorithm, args, file=args.file, tensorboard=args.tensorboard, numpy=args.numpy,
                                 model=args.model, buffer=args.buffer)

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

            if '-ram-' in self.env_name:  # Atari Ram state
                observation = observation / 255.

            done = False

            while not done:
                if self.render == True:
                    if self.domain_type in {'gym', "atari"}:
                        self.env.render()
                    elif self.domain_type in {'procgen'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array'))
                        cv2.waitKey(1)
                    elif self.domain_type in {'dmc', 'dmcr'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                action = self.algorithm.eval_action(observation)

                if self.discrete == False:
                    env_action = self.max_action * np.clip(action, -1, 1)
                else:
                    env_action = action

                next_observation, reward, done, _ = self.test_env.step(env_action)

                eval_reward += reward
                observation = next_observation

            reward_list.append(eval_reward)

        print("Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f} ".format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list)))

        if self.log == True:
            self.logger.log('Reward/Test', sum(reward_list)/len(reward_list), self.eval_num, False)
            self.logger.log('Max Reward/Test', max(reward_list), self.eval_num, False)
            self.logger.log('Min Reward/Test', min(reward_list), self.eval_num, False)
            self.logger.log('Stddev Reward/Test', np.std(reward_list), self.eval_num, True)


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

            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.render == True:
                    if self.domain_type in {'gym', "atari"}:
                        self.env.render()
                    elif self.domain_type in {'procgen'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array'))
                        cv2.waitKey(1)
                    elif self.domain_type in {'dmc', 'dmcr'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                if '-ram-' in self.env_name:  # Atari Ram state
                    observation = observation / 255.
                print(observation)

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
                    if self.log == True:
                        for loss in loss_list:
                            self.logger.log(loss[0], loss[1], self.total_step, str(self.episode))

                if self.eval == True and self.total_step % self.eval_step == 0:
                    self.evaluate()

                if self.log == True:
                    if self.model == True and self.total_step % self.model_freq == 0:
                        self.logger.save_model(self.algorithm, step=self.total_step)

                    if self.buffer == True and self.total_step % self.buffer_freq == 0:
                        self.logger.save_buffer(buffer=self.algorithm.buffer, step=self.total_step)

            print("Train | Episode: {}, Reward: {:.2f}, Local_step: {}, Total_step: {},".format(self.episode, self.episode_reward, self.local_step, self.total_step))

            if self.log == True:
                self.logger.log('Reward/Train', self.episode_reward, self.episode, False)
                self.logger.log('Step/Train', self.local_step, self.episode, False)
                self.logger.log('Total Step/Train', self.total_step, self.episode, True)





