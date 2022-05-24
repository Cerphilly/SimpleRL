import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import warnings
from Common.Utils import remove_argument, modify_choices, modify_default

def basic_config():
    parser = argparse.ArgumentParser(description='cocelRL Reinforcement Learning Experiment')
    #environment
    parser.add_argument('--domain_type', default='gym', type=str, choices=['gym', 'dmc', 'atari', 'procgen'], help='RL environment type: gym, dmc2gym(dm_control), atari(gym atari), dmc_image(dmc by image), procgen')
    parser.add_argument('--env-name', default='CartPole-v0', help='RL environment name')
    parser.add_argument('--render', default=False, action='store_true', help='render env state')
    #training
    parser.add_argument('--training-start', default=100, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=1000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #rl
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size (1e7 for state RL, 1e5 for visual RL)')
    parser.add_argument('--train-mode', default='offline', choices=['offline', 'online'], help='offline, online')
    parser.add_argument('--training-step', default=100, type=int, help='number of steps to train for each training. 1 if online')
    parser.add_argument('--gamma', default=0.99, type=float, help='Reward discount hyperparameter')

    #learning rate
    parser.add_argument('--learning-rate', default=0.001, type=float, help='Optimizer Learning rate')
    parser.add_argument('--actor-lr', default=0.001, type=float, help='Actor network learning rate')
    parser.add_argument('--critic-lr', default=0.001, type=float, help='Critic network learning rate')
    parser.add_argument('--v-lr', default=0.001, type=float, help='Value network learning rate')

    #network configuration
    parser.add_argument('--hidden-units', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--activation', default='relu', help='activation function of network')
    parser.add_argument('--use-bias', default=True, type=bool)
    parser.add_argument('--kernel-initializer', default='orthogonal')
    parser.add_argument('--bias-initializer', default='zeros')

    #setting & logging
    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=True, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--file', default=True, type=bool, help='when logged, write log')

    parser.add_argument('--save-model', default=True, type=bool, help='when logged, save model')
    parser.add_argument('--save-buffer', default=False, type=bool, help='when logged, save buffer')

    return parser

def image_config():

    parser = basic_config()

    parser.add_argument('--frame-stack', default=3, type=int)
    parser.add_argument('--frame-skip', default=8, type=int)
    parser.add_argument('--image-size', default=84, type=int)

    parser.add_argument('--layer-num', default=4, type=int)
    parser.add_argument('--filter-num', default=32, type=int)
    parser.add_argument('--feature-dim', default=50, type=int)
    parser.add_argument('--encoder-tau', default=0.05, type=float)
    modify_default(parser, 'buffer_size', 100000)

    return parser

def on_policy_config(parser=None):
    if parser == None:
        parser = basic_config()

    modify_default(parser, 'train_mode', 'offline')
    modify_choices(parser, 'train_mode', ['offline'])
    modify_default(parser, 'training_start', 0)
    modify_default(parser, 'training_step', 1)
    modify_default(parser, 'buffer_size', 1001)
    modify_default(parser, 'batch_size', 64)

    return parser




if __name__ == '__main__':
    parse = on_policy_config()
    print(parse.print_help())
    args = parse.parse_args()
    print(args)


