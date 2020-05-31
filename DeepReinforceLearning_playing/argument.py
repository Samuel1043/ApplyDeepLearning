import argparse

def add_arguments(parser):

    parser.add_argument('--double_dqn', default=False, help='whether using double DQN')
    parser.add_argument('--cpt_dir', default='./',help='checkpoint model for testing')
    parser.add_argument('--dueling_dqn', default=False, help='whether using dueling DQN')
    parser.add_argument('--baseline_pg', default=False, help='whether using dueling DQN')

    return parser