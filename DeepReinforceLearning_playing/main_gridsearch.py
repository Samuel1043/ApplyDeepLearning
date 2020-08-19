"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment
import os

def parse():
    parser = argparse.ArgumentParser(description="ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name or 'LunarLander-v2'
        env = Environment(env_name, args, atari_wrapper=False)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.train()

    if args.train_dqn:

        # train_freq = [1,4,16]
        # target_update_freq=[500,1000,3000]
        # buffer_size=[10000,20000,50000]
        # EPS_DECAY =[200,2000,10000]
        # EPS_START =[0.9]
        # EPS_END =[0.05]
        # gamma =[0.99,0.9,0.7]
        # batch_size=[16,32,64,128]

        # env_name = args.env_name or 'MsPacmanNoFrameskip-v0'
        # env = Environment(env_name, args, atari_wrapper=True)
        # from agent_dir.agent_dqn import AgentDQN

        # for i in train_freq:
        #     for j in target_update_freq:
        #         for k in EPS_DECAY:
        #             for l in gamma:
        #                 dir_name='%s_%s_%s_%s'%(str(i),str(j),str(k),str(l))
        #                 agent = AgentDQN(env, args,train_freq=i,target_update_freq=j,EPS_DECAY=k,GAMMA=l,dir_name=dir_name)
                        
        #                 os.system("mkdir ./save/dqn/"+dir_name)
        #                 agent.train(dir_name)

        env_name = args.env_name or 'MsPacmanNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)    
        agent.train()

    if args.test_pg:
        env = Environment('LunarLander-v2', args, test=True)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        test(agent, env, total_episodes=30)

    if args.test_dqn:
        env = Environment('MsPacmanNoFrameskip-v0', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=100)

if __name__ == '__main__':
    args = parse()
    run(args)
