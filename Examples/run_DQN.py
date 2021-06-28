import gym, dmc2gym
import argparse
import tensorflow as tf
import numpy as np
import random

from Algorithms.DQN import DQN

from Trainer.State_trainer import State_trainer

def hyperparameters():
    parser = argparse.ArgumentParser(description='Deep Q Network(DQN) example')
    #environment
    parser.add_argument('--domain_type', default='dmc', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='CartPole-v0', help='CartPole-v0, MountainCar-v0, Acrobot-v1, and atari games(not yet)')
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--training-start', default=100, type=int, help='First step to start training')
    parser.add_argument('--max-episode', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=1000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #dqn
    parser.add_argument('--batch-size', default=128, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='Offline, Online')
    parser.add_argument('--training-step', default=100, type=int)
    parser.add_argument('--copy-iter', default=2, type=int, help='Frequency to update target network')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float, help='Exploration probability')
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=True, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--file', default=False, type=bool, help='when logged, write log')
    parser.add_argument('--numpy', default=False, type=bool, help='when logged, save log in numpy')

    parser.add_argument('--model', default=False, type=bool, help='when logged, save model')
    parser.add_argument('--model-freq', default=10000, type=int, help='model saving frequency')
    parser.add_argument('--buffer', default=False, type=bool, help='when logged, save buffer')
    parser.add_argument('--buffer-freq', default=10000, type=int, help='buffer saving frequency')


    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')

    # random seed setting
    if args.random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = args.random_seed

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    #env setting
    if len(args.env_name.split('/')) == 1:
        #openai gym
        env = gym.make(args.env_name)
        env.seed(random_seed)
        env.action_space.seed(random_seed)

        test_env = gym.make(args.env_name)
        test_env.seed(random_seed)
        test_env.action_space.seed(random_seed)
    else:
        #deepmind control suite
        env = dmc2gym.make(domain_name=args.env_name.split('/')[0], task_name=args.env_name.split('/')[1], seed=random_seed)
        test_env = dmc2gym.make(domain_name=args.env_name.split('/')[0], task_name=args.env_name.split('/')[1], seed=random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_action = 1
    min_action = 1

    algorithm = DQN(state_dim, action_dim, args)

    print("Training of", env.unwrapped.spec.id)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    trainer = State_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

