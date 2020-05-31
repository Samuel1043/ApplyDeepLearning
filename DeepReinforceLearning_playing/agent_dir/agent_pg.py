import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent_dir.agent import Agent
from environment import Environment

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.save_pth=args.cpt_dir+'pg'
        self.args=args
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load(self.save_pth+'.cpt')

        if self.args.baseline_pg:
            self.save_pth+='_baseline'
        # discounted reward
        self.gamma = 0.99
        self.eps=1e-8

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
 
        # saved rewards and actions add
        self.rewards, self.saved_log_probs = [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_log_probs = [], []

    def make_action(self, state, test=False):
        if test:
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs=self.model(state)
            m = Categorical(probs)
            action = m.sample()
            return action.item()
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs=self.model(state)
            m = Categorical(probs)
            action = m.sample()

            # Use your model to output distribution over actions and sample from it.
            # HINT: torch.distributions.Categorical
            return m.log_prob(action),action.item()

    def update(self):
        policy_loss = []
        returns = []
        R = 0

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        if self.args.baseline_pg:
            baseline_value=returns.sum(0)/returns.shape[0]

        for log_prob, R in zip(self.saved_log_probs, returns):
            if self.args.baseline_pg:
                policy_loss.append(-log_prob * (R-baseline_value))
            else:
                policy_loss.append(-log_prob * (R))

        policy_loss = torch.cat(policy_loss).sum()
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}

        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self):
        avg_reward = None
        f=open(self.save_pth+'_performance.txt','w')
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                saved_log_prob,action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.rewards.append(reward)
                self.saved_log_probs.append(saved_log_prob)
            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                f.write('Epochs: %d/%d | Avg reward: %f \n'%
                       (epoch, self.num_episodes, avg_reward))
            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                f.write('Epochs: %d/%d | Avg reward: %f \n'%
                       (epoch, self.num_episodes, avg_reward))
                self.save(self.save_pth+'.cpt')
                break
        f.close()
