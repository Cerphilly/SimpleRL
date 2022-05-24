import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Common.Utils import dmc_image_env, dmc_diff_image_env, env_info, cpu_only, set_seed
from Common.Config import image_config
from Algorithm.ImageRL.RAD import RAD_SACv2

from Trainer.Basic_trainer import Basic_trainer


def rad_configurations(parser):
    parser.set_defaults(domain_type='dmc_image')
    parser.set_defaults(env_name='cartpole_swingup')
    parser.set_defaults(render=False)
    parser.set_defaults(max_step=100001)
    parser.set_defaults(eval=True)
    parser.set_defaults(eval_step=10000)
    parser.set_defaults(eval_episode=10)
    parser.set_defaults(random_seed=37)
    parser.set_defaults(training_start=1000)
    parser.set_defaults(batch_size=512)
    parser.set_defaults(train_mode='online')
    parser.set_defaults(training_step=1)

    parser.set_defaults(data_aug='translate')
    parser.set_defaults(pre_image_size=100)
    parser.set_defaults(image_size=108)
    parser.set_defaults(hidden_units=(1024, 1024))
    parser.set_defaults(actor_lr=0.001)
    parser.set_defaults(critic_lr=0.001)
    parser.set_defaults(alpha_lr=0.0001)
    parser.set_defaults(tau=0.01)
    parser.set_defaults(alpha=0.1)
    parser.set_defaults(encoder_tau=0.05)
    parser.set_defaults(critic_update=2)

    parser.set_defaults(frame_stack=3)
    parser.set_defaults(frame_skip=8)

    parser.set_defaults(log=True)


    return parser


def main(args):
    if args.cpu_only == True:
        cpu_only()

    # random seed setting
    random_seed = set_seed(args.random_seed)

    # RAD:
        #crop: args.image_size < args.pre_image_size
        #translate: args.image_size > args.pre_image.size
    # env setting
    if args.domain_type == 'dmc_image':
        env, test_env = dmc_image_env(args.env_name, args.pre_image_size, args.frame_stack, args.frame_skip, True, random_seed)
        args.data_format = 'channels_first'

    elif args.domain_type == 'dmc_diff_image':
        env, test_env = dmc_diff_image_env(args.env_name, args.pre_image_size, args.frame_stack, args.frame_skip, True, random_seed)
        args.data_format = 'channels_first'

    else:
        raise ValueError("only dmc_image allowed")

    state_dim, action_dim, max_action, min_action = env_info(env)
    state_dim = (state_dim[0], args.image_size, args.image_size)#RAD

    algorithm = RAD_SACv2(state_dim, action_dim, args)

    trainer = Basic_trainer(env, test_env, algorithm, max_action, min_action, args)
    trainer.run()


if __name__ == '__main__':
    parser = image_config()
    parser = RAD_SACv2.get_config(parser)
    parser = rad_configurations(parser)
    args = parser.parse_args()
    main(args)
