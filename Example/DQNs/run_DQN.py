import argparse

from Algorithm.DQNs.DQN import DQN
from Algorithm.ImageRL.DQN import ImageDQN

from Trainer.Basic_trainer import Basic_trainer
from Common.Utils import cpu_only, set_seed, gym_env, atari_env,  procgen_env, print_envs, print_args, env_info

def hyperparameters():
    parser = argparse.ArgumentParser(description='Deep Q Network(DQN) example')
    #environment
    parser.add_argument('--domain_type', default='atari', type=str, help='gym or dmc or atari')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4', help='CartPole-v0, MountainCar-v0, Acrobot-v1, and atari games, PongNoFrameskip-v4')
    parser.add_argument('--render', default=True, type=bool)

    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=10000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=1, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #dqn
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=100000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='online', help='Offline, Online')
    parser.add_argument('--training-step', default=1, type=int)
    parser.add_argument('--copy-iter', default=100, type=int, help='Frequency to update target network')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float, help='Exploration probability')
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--activation', default='relu')
    #atari setting
    parser.add_argument('--frame-stack', default=3, type=int)
    parser.add_argument('--frame-skip', default=4, type=int)
    parser.add_argument('--image-size', default=84, type=int)

    parser.add_argument('--layer-num', default=4, type=int)
    parser.add_argument('--filter-num', default=32, type=int)
    parser.add_argument('--kernel-size', default=3, type=int)
    parser.add_argument('--strides', default=(2, 1, 1, 1))
    parser.add_argument('--feature-dim', default=50, type=int)

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

    elif args.domain_type == 'atari':
        env, test_env = atari_env(args.env_name, args.image_size, args.frame_stack, args.frame_skip, random_seed)

    elif args.domain_type == 'procgen':
        env, test_env = procgen_env(args.env_name, args.frame_stack, random_seed)

    else:
        raise ValueError

    state_dim, action_dim, max_action, min_action = env_info(env)

    if args.domain_type in {'gym'}:
        algorithm = DQN(state_dim, action_dim, args)

    elif args.domain_type in {'atari', 'procgen'}:
        algorithm = ImageDQN(state_dim, action_dim, args)

    else:
        raise ValueError

    print_args(args)
    print_envs(algorithm, max_action, min_action, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

