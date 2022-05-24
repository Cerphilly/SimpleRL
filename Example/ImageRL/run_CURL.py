import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Common.Utils import dmc_image_env, env_info, cpu_only, set_seed
from Common.Config import image_config
from Algorithm.ImageRL.CURL import CURL_SACv2

from Trainer.Basic_trainer import Basic_trainer

def curl_configurations(parser):
    parser.set_defaults(domain_type ='dmc_image')
    parser.set_defaults(env_name ='cartpole_swingup')
    parser.set_defaults(render = False)
    parser.set_defaults(eval = True)
    parser.set_defaults(eval_step = 10000)
    parser.set_defaults(eval_episode = 1)
    parser.set_defaults(random_seed = -1)
    parser.set_defaults(training_start = 1000)
    parser.set_defaults(batch_size = 256)
    parser.set_defaults(train_mode = 'online')
    parser.set_defaults(training_step = 1)

    parser.set_defaults(frame_stack=3)
    parser.set_defaults(frame_skip=8)

    return parser

def main(args):
    if args.cpu_only == True:
        cpu_only()

    # random seed setting
    random_seed = set_seed(args.random_seed)

    #env setting
    if args.domain_type == 'dmc_image':
        env, test_env = dmc_image_env(args.env_name, args.pre_image_size, args.frame_stack, args.frame_skip, random_seed)
        args.data_format = 'channels_first'

    else:
        raise ValueError("only dmc_image allowed")

    state_dim, action_dim, max_action, min_action = env_info(env)

    algorithm = CURL_SACv2(state_dim, action_dim, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    parser = image_config()
    parser = CURL_SACv2.get_config(parser)
    parser = curl_configurations(parser)
    args = parser.parse_args()
    print(args)
    main(args)

