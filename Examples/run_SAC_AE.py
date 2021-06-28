import dmc2gym, gym
import argparse
import tensorflow as tf
import numpy as np
import random

from Algorithms.ImageRL.SAC_AE import SACv2_AE

from Trainer.Image_trainer import Image_trainer
from Common.Utils import FrameStack

def hyperparameters():
    parser = argparse.ArgumentParser(description='Soft Actor Critic with AutoEncoder (SAC_AE) example')
    #environment
    parser.add_argument('--algorithm', default='SACv2', help='SACv2')
    parser.add_argument('--domain_type', default='dmc', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='cartpole/swingup', help='DM Control Suite domain name + task name')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-episode', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=5, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    parser.add_argument('--frame-stack', default=3, type=int)
    parser.add_argument('--frame-skip', default=8, type=int)
    parser.add_argument('--image-size', default=84, type=int)

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
    parser.add_argument('--actor-update', default=2, type=int)
    parser.add_argument('--critic-update', default=2, type=int)
    parser.add_argument('--decoder-update', default=1, type=int)

    parser.add_argument('--hidden-dim', default=(1024, 1024), help='hidden dimension of network')
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')

    #encoder & decoder
    parser.add_argument('--layer-num', default=4, type=int)
    parser.add_argument('--filter-num', default=32, type=int)
    parser.add_argument('--encoder-tau', default=0.05, type=float)
    parser.add_argument('--encoder-lr', default=0.001, type=float)
    parser.add_argument('--feature-dim', default=50, type=int)
    parser.add_argument('--decoder-lr', default=0.001, type=float)
    parser.add_argument('--decoder-latent-lambda', default=1e-6, type=float)
    parser.add_argument('--decoder-weight-lambda', default=1e-7, type=float)

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

    domain_name = args.env_name.split('/')[0]
    task_name = args.env_name.split('/')[1]
    env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True,
                       height=args.image_size, width=args.image_size, frame_skip=args.frame_skip)#Pre image size for curl, image size for dbc
    env = FrameStack(env, k=args.frame_stack)

    test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True,
                       height=args.image_size, width=args.image_size, frame_skip=args.frame_skip)#Pre image size for curl, image size for dbc
    test_env = FrameStack(test_env, k=args.frame_stack)

    state_dim = (3 * args.frame_stack, args.image_size, args.image_size)
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]

    if args.algorithm == 'SACv1':
        algorithm = 0
    elif args.algorithm == 'SACv2':
        algorithm = SACv2_AE(state_dim, action_dim, args)


    print("Training of", env.unwrapped.spec.id)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)

    trainer = Image_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

