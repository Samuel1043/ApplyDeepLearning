import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple
from agent_dir.agent import Agent
from environment import Environment
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q

class DuelingDQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #Value function estimates how good the state is
        self.value = nn.Sequential(nn.Linear(3136, 512),nn.Linear(512, 1))
        #Advantage function estimates the additional benefit
        self.action = nn.Sequential(nn.Linear(3136, 512),nn.Linear(512, num_actions))

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        v=self.lrelu(self.value(x.view(x.size(0), -1)))
        a=self.lrelu(self.action(x.view(x.size(0), -1)))
        
        #setting sum of a to zero to make v changes more
        q= v+a-a.mean(1).unsqueeze(1)
        return q

class AgentDQN(Agent):
    def __init__(self, env, args,**grid_param):
        self.env = env
        self.input_channels = 4
        self.num_actions  = self.env.action_space.n

        self.save_dir=args.cpt_dir+'dqn'
        self.double_dqn=args.double_dqn
        self.dueling_dqn=args.dueling_dqn

        if self.double_dqn:
            self.save_dir=self.save_dir+'double_'
        

        # build target, online network
        if self.dueling_dqn:
            self.save_dir=self.save_dir+'duel_'
            self.target_net = DuelingDQN(self.input_channels, self.num_actions)
            self.target_net = self.target_net.cuda() if use_cuda else self.target_net
            self.online_net = DuelingDQN(self.input_channels, self.num_actions)
            self.online_net = self.online_net.cuda() if use_cuda else self.online_net
        else:
            self.target_net = DQN(self.input_channels, self.num_actions)
            self.target_net = self.target_net.cuda() if use_cuda else self.target_net
            self.online_net = DQN(self.input_channels, self.num_actions)
            self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        

        if args.test_dqn:
            self.load(self.save_dir)

        # discounted reward
        self.GAMMA = 0.99

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 1000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 100000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network
        self.buffer_size = 10000 # max size of replay buffer

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

        self.steps = 0 # num. of passed steps

        # TODO: initialize your replay buffer
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200

        self.memory=ReplayMemory(self.buffer_size)


        
        # grid search parameters
        # self.train_freq=grid_param['train_freq']
        # self.target_update_freq=grid_param['target_update_freq']
        # self.EPS_DECAY=grid_param['EPS_DECAY']
        # self.GAMMA=grid_param['GAMMA']


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        if test:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(device)
            return self.online_net(state).max(1)[1].item()
        else:
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps / self.EPS_DECAY)
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.online_net(state).max(1)[1].view(1,1)
            else:
                return torch.tensor([[self.env.action_space.sample()]]).to(device)

    def update(self):
        # TODO:
        # step 1: Sample some stored experiences as training examples.
        # step 2: Compute Q(s_t, a) with your model.
        # step 3: Compute Q(s_{t+1}, a) with target model.
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # step 5: Compute temporal difference loss
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.
        experiences=self.memory.sample(self.batch_size)
        batch = Transition(*zip(*experiences))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)


        state_values = self.online_net(state_batch)
        state_action_values =state_values.gather(1, action_batch).squeeze(1)


        next_state_values = torch.zeros(self.batch_size, device=device)


        if self.double_dqn:
            with torch.no_grad():
                online_action=state_values.argmax(1)[non_final_mask]
                next_state_values[non_final_mask]=self.target_net(non_final_next_states)[non_final_mask,online_action].detach()
        else:

            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        loss=self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        w=open(self.save_dir+'performance.txt','w')
        while(True):
                state = self.env.reset()
                # State: (80,80,4) --> (1,4,80,80)
                state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(device)

                done = False
                while(not done):
                    # select and perform action
                    action = self.make_action(state)
                    next_state, reward, done, _ = self.env.step(action.item())
                    reward=torch.tensor([reward]).to(device)

                    total_reward += reward.item()

                    # process new state
                    next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0).to(device)

                    # TODO: store the transition in memory
                    self.memory.push(state,action,next_state,reward)
                    # move to the next state
                    state = next_state
                    # Perform one step of the optimization
                    if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                        loss = self.update()

                    # TODO: update target network
                    if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                        self.target_net.load_state_dict(self.online_net.state_dict())

                    # save the model
                    if self.steps % self.save_freq == 0:
                        self.save(self.save_dir)

                    self.steps += 1

                if episodes_done_num % self.display_freq == 0:
                    print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                            (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                    w.write('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f \n'%
                            (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))

                    total_reward = 0

                episodes_done_num += 1
                if self.steps > self.num_timesteps:
                    break
        w.close()
        self.save(self.save_dir)








    # different hyperparameter
    
    # def train(self,dir_name):


    #     dir_name=dir_name

    #     episodes_done_num = 0 # passed episodes
    #     total_reward = 0 # compute average reward
    #     loss = 0
    #     f=open('./save/dqn/%s/train_progress.txt'%(dir_name),'w')
    #     while(True):
            
    #         state = self.env.reset()
    #         # State: (80,80,4) --> (1,4,80,80)
    #         state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(device)

    #         done = False
    #         while(not done):
    #             # select and perform action
    #             action = self.make_action(state)
    #             next_state, reward, done, _ = self.env.step(action.item())
    #             reward=torch.tensor([reward]).to(device)

    #             total_reward += reward.item()

    #             # process new state
    #             next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0).to(device)

    #             # TODO: store the transition in memory
    #             self.memory.push(state,action,next_state,reward)
    #             # move to the next state
    #             state = next_state
    #             # Perform one step of the optimization
    #             if self.steps > self.learning_start and self.steps % self.train_freq == 0:
    #                 loss = self.update()

    #             # TODO: update target network
    #             if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
    #                 self.target_net.load_state_dict(self.online_net.state_dict())

    #             # save the model
    #             if self.steps % self.save_freq == 0:
    #                 self.save('./save/dqn/%s/dqn'%(dir_name))

    #             self.steps += 1

    #         if episodes_done_num % self.display_freq == 0:
    #             print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
    #                     (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
    #             f.write('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f\n'%
    #                     (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))

    #             total_reward = 0

    #         episodes_done_num += 1
    #         if self.steps > self.num_timesteps:
    #             f.close()
    #             break
    #     self.save('./save/dqn/%s/dqn'%(dir_name))




