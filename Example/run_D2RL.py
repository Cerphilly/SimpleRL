import argparse
import tensorflow as tf
import numpy as np
import random

from Algorithm.D2RL import D2RL_TD3, D2RL_SAC_v2, D2RL_SAC_v1
from Common.Utils import cpu_only, set_seed, gym_env, dmc_env, print_envs, print_args, env_info
from Trainer.Basic_trainer import Basic_trainer

def hyperparameters():
    parser = argparse.ArgumentParser(description='D2RL example')
    #environment
    parser.add_argument('--algorithm', default='SACv1', help='SACv1, SACv2, TD3')
    parser.add_argument('--domain_type', default='gym', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='InvertedPendulumSwing-v2', help='env name')
    parser.add_argument('--discrete', default=False, type=bool, help='Always Continuous')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')

    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    #sac
    parser.add_argument('--batch-size', default=128, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=100000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='online', help='offline, online')
    parser.add_argument('--training-step', default=1, type=int)
    parser.add_argument('--train-alpha', default=True, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--v-lr', default=0.001, type=float)
    parser.add_argument('--alpha-lr', default=0.0001, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--activation', default='relu')

    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')
    #td3
    parser.add_argument('--policy-delay', default=2, type=int)
    parser.add_argument('--actor-noise', default=0.1, type=float)
    parser.add_argument('--target-noise', default=0.2, type=float)
    parser.add_argument('--noise-clip', default=0.5, type=float)

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')


    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only == True:
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

    if args.algorithm == 'SACv1':
        algorithm = D2RL_SAC_v1(state_dim, action_dim, args)
    elif args.algorithm == 'SACv2':
        algorithm = D2RL_SAC_v2(state_dim, action_dim, args)
    elif args.algorithm == 'TD3':
        algorithm = D2RL_TD3(state_dim, action_dim, args)
    else:
        raise NotImplementedError

    print_args(args)
    print_envs(algorithm, max_action, min_action, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

