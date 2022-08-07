import argparse

from Algorithm.ImageRL.CURL import CURL_SACv1, CURL_TD3, CURL_SACv2

from Trainer.Basic_trainer import Basic_trainer
from Common.Utils import cpu_only, set_seed, dmc_image_env, dmcr_env, print_envs, print_args, env_info

def hyperparameters():
    parser = argparse.ArgumentParser(description='Contrastive Unsupervised Representations for Reinforcement Learning (CURL) example')
    #environment
    parser.add_argument('--algorithm', default='SACv2', help='SACv1, SACv2, TD3')
    parser.add_argument('--domain_type', default='dmc_image', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='cartpole_swingup', help='DM Control Suite domain name + task name')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=100001, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=1000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=10, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=1234, type=int, help='Random seed setting')

    parser.add_argument('--frame-stack', default=3, type=int)
    parser.add_argument('--frame-skip', default=8, type=int)
    parser.add_argument('--image-size', default=84, type=int)
    parser.add_argument('--pre-image-size', default=100, type=int)
    #sac
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=100000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='online', help='online')
    parser.add_argument('--training-step', default=1, type=int)
    parser.add_argument('--train-alpha', default=True, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--v-lr', default=0.001, type=float)
    parser.add_argument('--alpha-lr', default=0.0001, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--critic-update', default=2, type=int)
    parser.add_argument('--hidden-dim', default=(1024, 1024), help='hidden dimension of network')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')
    #td3
    parser.add_argument('--policy-delay', default=2, type=int)
    parser.add_argument('--actor-noise', default=0.1, type=float)
    parser.add_argument('--target-noise', default=0.2, type=float)
    parser.add_argument('--noise-clip', default=0.5, type=float)
    #curl&encoder
    parser.add_argument('--layer-num', default=4, type=int)
    parser.add_argument('--filter-num', default=32, type=int)
    parser.add_argument('--kernel-size', default=3, type=int)
    parser.add_argument('--strides', default=(2, 1, 1, 1))

    parser.add_argument('--encoder-tau', default=0.05, type=float)

    parser.add_argument('--feature-dim', default=50, type=int)
    parser.add_argument('--curl-latent-dim', default=128, type=int)
    parser.add_argument('--encoder-lr', default=0.001, type=float)
    parser.add_argument('--cpc-lr', default=0.001, type=float)

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')

    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only:
        cpu_only()
    # random seed setting
    random_seed = set_seed(args.random_seed)

    if args.domain_type == 'dmc_image':
        env, test_env = dmc_image_env(args.env_name, args.pre_image_size, args.frame_stack, args.frame_skip, random_seed)

    elif args.domain_type == 'dmcr':
        env, test_env = dmcr_env(args.env_name, args.pre_image_size, args.frame_skip, random_seed, mode='classic')
    else:
        raise ValueError

    state_dim, action_dim, max_action, min_action = env_info(env)
    state_dim = (state_dim[0], args.image_size, args.image_size)

    if args.algorithm == 'SACv1':
        algorithm = CURL_SACv1(state_dim, action_dim, args)
    elif args.algorithm == 'SACv2':
        algorithm = CURL_SACv2(state_dim, action_dim, args)
    elif args.algorithm == 'TD3':
        algorithm = CURL_TD3(state_dim, action_dim, args)
    else:
        raise ValueError

    print_args(args)
    print_envs(algorithm, max_action, min_action, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

