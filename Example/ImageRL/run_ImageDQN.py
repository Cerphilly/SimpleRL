import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Common.Utils import atari_env, procgen_env, env_info, cpu_only, set_seed
from Common.Config import image_config
from Common.Logger import print_args

from Algorithm.ImageRL.DQN import ImageDQN

from Trainer.Basic_trainer import Basic_trainer

def image_dqn_configurations(parser):
    parser.set_defaults(domain_type ='atari')
    parser.set_defaults(env_name ='PongNoFrameskip-v4')
    parser.set_defaults(render = True)
    parser.set_defaults(eval = True)
    parser.set_defaults(eval_step = 10000)
    parser.set_defaults(eval_episode = 1)
    parser.set_defaults(random_seed = -1)
    parser.set_defaults(training_start = 200)
    parser.set_defaults(batch_size = 256)
    parser.set_defaults(train_mode = 'online')
    parser.set_defaults(training_step = 1)
    parser.set_defaults(copy_iter = 100)
    parser.set_defaults(epsilon = 0.1)
    parser.set_defaults(frame_skip=4)
    return parser

def main(args):
    if args.cpu_only == True:
        cpu_only()

    # random seed setting
    random_seed = set_seed(args.random_seed)

    #env setting
    if args.domain_type == 'atari':
        env, test_env = atari_env(args.env_name, args.image_size, args.frame_stack, args.frame_skip, random_seed)

        args.data_format = 'channels_first'

    elif args.domain_type == 'procgen':
        env, test_env = procgen_env(args.env_name, args.frame_stack)
        args.data_format = 'channels_last'

    else:
        raise ValueError("only atari, procgen allowed")

    state_dim, action_dim, max_action, min_action = env_info(env)

    algorithm = ImageDQN(state_dim, action_dim, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    parser = image_config()
    parser = ImageDQN.get_config(parser)
    parser = image_dqn_configurations(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    print_args(args)
    main(args)

