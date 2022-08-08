import argparse
import tensorflow as tf
import random
import numpy as np

from Algorithm.TD3 import TD3

from Common.Utils import cpu_only, set_seed, gym_env, dmc_env, env_info, print_envs, print_args
from Trainer.Basic_trainer import Basic_trainer

def hyperparameters():
    parser = argparse.ArgumentParser(description='Deep Delayed Deterministic Policy Gradient(TD3) example')
    #environment
    parser.add_argument('--domain_type', default='dmc', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='cartpole_swingup', help='Pendulum-v0, MountainCarContinuous-v0')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=200, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #ddpg
    parser.add_argument('--batch-size', default=128, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='offline, online')
    parser.add_argument('--training_step', default=200, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--policy-delay', default=2, type=int)
    parser.add_argument('--actor-noise', default=0.1, type=float)
    parser.add_argument('--target-noise', default=0.2, type=float)
    parser.add_argument('--noise-clip', default=0.5, type=float)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--activation', default='relu')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')

    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only:
        cpu_only()

    # random seed setting
    random_seed = set_seed(args.random_seed)

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc':
        env, test_env = dmc_env(args.env_name, random_seed)

    else:
        raise ValueError

    state_dim, action_dim, max_action, min_action = env_info(env)
    algorithm = TD3(state_dim, action_dim, args)

    print_args(args)
    print_envs(algorithm, max_action, min_action, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

