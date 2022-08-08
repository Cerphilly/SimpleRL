import argparse


from Algorithm.SAC_v2 import SAC_v2
from Algorithm.ImageRL.SAC import ImageSAC_v2

from Trainer.Basic_trainer import Basic_trainer
from Common.Utils import cpu_only, set_seed, gym_env, dmc_env, dmc_image_env, dmcr_env, env_info, print_envs, print_args

def hyperparameters():
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v2 example')
    #environment
    parser.add_argument('--domain_type', default='gym', type=str, help='gym or dmc, dmc/image')
    parser.add_argument('--env-name', default='InvertedPendulumSwing-v2', help='Pendulum-v0, MountainCarContinuous-v0')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=100001, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=10, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=1234, type=int, help='Random seed setting')
    #sac
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='offline, online')
    parser.add_argument('--training-step', default=256, type=int)
    parser.add_argument('--train-alpha', default=True, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--alpha-lr', default=0.0001, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--critic-update', default=1, type=int)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')
    #image
    parser.add_argument('--frame-stack', default=3, type=int)
    parser.add_argument('--frame-skip', default=8, type=int)
    parser.add_argument('--image-size', default=84, type=int)

    parser.add_argument('--layer-num', default=4, type=int)
    parser.add_argument('--filter-num', default=32, type=int)
    parser.add_argument('--kernel-size', default=3, type=int)
    parser.add_argument('--strides', default=(2, 1, 1, 1))

    parser.add_argument('--feature-dim', default=50, type=int)
    parser.add_argument('--encoder-tau', default=0.05, type=float)

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

    elif args.domain_type == 'dmc_image':
        env, test_env = dmc_image_env(args.env_name, args.image_size, args.frame_stack, args.frame_skip, random_seed)

    elif args.domain_type == 'dmcr':
        env, test_env = dmcr_env(args.env_name, args.image_size, args.frame_skip, random_seed, 'sim2real')
    else:
        raise ValueError

    state_dim, action_dim, max_action, min_action = env_info(env)

    if args.domain_type in {'gym', 'dmc'}:
        algorithm = SAC_v2(state_dim, action_dim, args)

    elif args.domain_type in {'dmc_image', 'dmcr'}:
        algorithm = ImageSAC_v2(state_dim, action_dim, args)

    else:
        raise ValueError

    print_args(args)
    print_envs(algorithm, max_action, min_action, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

