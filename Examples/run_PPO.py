import argparse
import tensorflow as tf
import numpy as np
import random

from Algorithms.PPO import PPO
from Algorithms.ImageRL.PPO import ImagePPO
from Common.Utils import FrameStack
from Trainer.On_policy_trainer import On_policy_trainer

def hyperparameters():
    parser = argparse.ArgumentParser(description='Proximal Policy Gradient(PPO) example')
    #environment
    parser.add_argument('--domain_type', default='procgen', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='starpilot', help='Pendulum-v0, MountainCarContinuous-v0, CartPole-v0')
    parser.add_argument('--discrete', default=True, type=bool, help='whether the environment is discrete or not')
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--training-start', default=0, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=False, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=10, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #ppo
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='offline')
    parser.add_argument('--ppo-mode', default='clip', help='Clip, Adaptive KL, Fixed KL')
    parser.add_argument('--clip', default=0.2, type=float)
    parser.add_argument('--training-step', default=1, type=int, help='inverteddobulependulum-v2: 1')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda-gae', default=0.95, type=float)
    parser.add_argument('--actor-lr', default=0.0003, type=float)
    parser.add_argument('--critic-lr', default=0.0003, type=float)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')

    parser.add_argument('--frame-stack', default=3, type=int)
    parser.add_argument('--frame-skip', default=4, type=int)
    parser.add_argument('--image-size', default=84, type=int)
    parser.add_argument('--layer-num', default=4, type=int)
    parser.add_argument('--filter-num', default=32, type=int)
    parser.add_argument('--feature-dim', default=50, type=int)

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=False, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--file', default=True, type=bool, help='when logged, write log')
    parser.add_argument('--numpy', default=True, type=bool, help='when logged, save log in numpy')

    parser.add_argument('--model', default=False, type=bool, help='when logged, save model')
    parser.add_argument('--model-freq', default=10000, type=int, help='model saving frequency')

    parser.add_argument('--buffer', default=False, type=bool, help='when logged, save buffer')
    parser.add_argument('--buffer-freq', default=100000, type=int, help='buffer saving frequency')

    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only == True:
        cpu = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=cpu, device_type='CPU')
        tf.config.set_visible_devices([], 'GPU')

    # random seed setting
    if args.random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = args.random_seed

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    #env setting
    if args.domain_type == 'gym':
        import gym
        #openai gym
        env = gym.make(args.env_name)
        env.seed(random_seed)
        env.action_space.seed(random_seed)

        test_env = gym.make(args.env_name)
        test_env.seed(random_seed)
        test_env.action_space.seed(random_seed)

    elif args.domain_type == 'dmc':
        import dmc2gym
        #deepmind control suite
        env = dmc2gym.make(domain_name=args.env_name.split('/')[0], task_name=args.env_name.split('/')[1], seed=random_seed)
        test_env = dmc2gym.make(domain_name=args.env_name.split('/')[0], task_name=args.env_name.split('/')[1], seed=random_seed)

    elif args.domain_type == 'dmc/image':
        import dmc2gym
        domain_name = args.env_name.split('/')[0]
        task_name = args.env_name.split('/')[1]
        env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True, height=args.image_size, width=args.image_size, frame_skip=args.frame_skip)#Pre image size for curl, image size for dbc
        env = FrameStack(env, k=args.frame_stack)

        test_env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=random_seed, visualize_reward=False, from_pixels=True, height=args.image_size, width=args.image_size, frame_skip=args.frame_skip)#Pre image size for curl, image size for dbc
        test_env = FrameStack(test_env, k=args.frame_stack)

    elif args.domain_type == 'dmcr':
        import dmc_remastered as dmcr
        domain_name = args.env_name.split('/')[0]
        task_name = args.env_name.split('/')[1]

        env, test_env = dmcr.benchmarks.classic(domain_name, task_name, visual_seed=0, width=args.image_size, height=args.image_size, frame_skip=args.frame_skip)
        # env, test_env = dmcr.benchmarks.visual_generalization(domain_name, task_name, num_levels=100, width=args.pre_image_size, height=args.pre_image_size, frame_skip=args.frame_skip)
        # env, test_env = dmcr.benchmarks.visual_sim2real(domain_name, task_name, num_levels=100, width=args.pre_image_size, height=args.pre_image_size, frame_skip=args.frame_skip)

    elif args.domain_type == 'procgen':
        import gym
        env_name = "procgen:procgen-{}-v0".format(args.env_name)
        env = gym.make(env_name, render_mode='rgb_array')
        env._max_episode_steps = 1000
        env = FrameStack(env, args.frame_stack, data_format='channels_last')

        test_env = gym.make(env_name, render_mode='rgb_array')
        test_env._max_episode_steps = 1000
        test_env = FrameStack(test_env, args.frame_stack, data_format='channels_last')

    if args.discrete == True:
        state_dim = env.observation_space.shape[0]
        if args.domain_type in {'atari', 'procgen'}:
            state_dim = env.observation_space.shape

        action_dim = env.action_space.n
        max_action = 1
        min_action = 1
    else:
        state_dim = env.observation_space.shape[0]
        if args.domain_type in {'dmc/image', 'dmcr'}:
            state_dim = env.observation_space.shape

        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        min_action = env.action_space.low[0]

    if args.domain_type in {'gym', 'dmc'}:
        algorithm = PPO(state_dim, action_dim, args)

    elif args.domain_type in {'atari', 'procgen', 'dmc/image', 'dmcr'}:
        algorithm = ImagePPO(state_dim, action_dim, args)

    print("Training of", args.domain_name + '_' + args.task_name)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)
    print("Discrete: ", args.discrete)

    trainer = On_policy_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

