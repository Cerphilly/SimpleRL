import argparse


from Algorithm.PPO import PPO
from Algorithm.ImageRL.PPO import ImagePPO
from Common.Utils import cpu_only, set_seed, gym_env, dmc_env, dmc_image_env, dmcr_env, procgen_env
from Trainer.Basic_trainer import Basic_trainer

def hyperparameters():
    parser = argparse.ArgumentParser(description='Proximal Policy Gradient(PPO) example')
    #environment
    parser.add_argument('--domain_type', default='dmc', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='cartpole_swingup', help='Pendulum-v0, MountainCarContinuous-v0, CartPole-v0')
    parser.add_argument('--discrete', default=False, type=bool, help='whether the environment is discrete or not')
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--training-start', default=0, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=1000000, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=False, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=10, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #ppo
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--buffer-size', default=1000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='offline')
    parser.add_argument('--ppo-mode', default='clip', help='Clip, Adaptive KL, Fixed KL')
    parser.add_argument('--clip', default=0.2, type=float)
    parser.add_argument('--training-step', default=1, type=int, help='inverteddobulependulum-v2: 1')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda-gae', default=0.95, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
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

    parser.add_argument('--model', default=True, type=bool, help='when logged, save model')
    parser.add_argument('--model-freq', default=100000, type=int, help='model saving frequency')

    parser.add_argument('--buffer', default=False, type=bool, help='when logged, save buffer')
    parser.add_argument('--buffer-freq', default=100000, type=int, help='buffer saving frequency')

    args = parser.parse_args()

    return args

def main(args):
    if args.cpu_only == True:
        cpu_only()

    random_seed = set_seed(args.random_seed)

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc':
        env, test_env = dmc_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc_image':
        env, test_env = dmc_image_env(args.env_name, args.image_size, args.frame_stack, args.frame_skip, random_seed)

    elif args.domain_type == 'dmcr':
        env, test_env = dmcr_env(args.env_name, args.image_size, args.frame_skip, random_seed, mode='classic')

    elif args.domain_type == 'procgen':
        env, test_env = procgen_env(args.env_name, args.frame_stack, random_seed)


    if args.discrete == True:
        state_dim = env.observation_space.shape[0]

        if args.domain_type in {'atari', 'procgen'}:
            state_dim = env.observation_space.shape

        action_dim = env.action_space.n
        max_action = 1
        min_action = 1
    else:
        state_dim = env.observation_space.shape[0]

        if args.domain_type in {'dmc_image', 'dmcr'}:
            state_dim = env.observation_space.shape

        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        min_action = env.action_space.low[0]


    if args.domain_type in {'gym', 'dmc'}:
        algorithm = PPO(state_dim, action_dim, args)

    elif args.domain_type in {'atari', 'procgen', 'dmc_image', 'dmcr'}:
        algorithm = ImagePPO(state_dim, action_dim, args)

    print("Training of", args.domain_type + '_' + args.env_name)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Min action:", min_action)
    print("Discrete: ", args.discrete)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

