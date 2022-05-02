import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import heapq
import random

import matplotlib
import matplotlib.pyplot as plt
import itertools
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
use_cuda =  torch.cuda.is_available()
FloatTensor =  torch.FloatTensor
LongTensor =  torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


class ReplayMemory(object):

    def __init__(self, capacity=500):
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

class Policy(nn.Module):
    def __init__(self, n_hotel, n_dyn):
        super(Policy, self).__init__()
        self.dim = n_hotel+365
        self.rnn = nn.LSTM(self.dim,self.dim,num_layers=2, dropout=0.5)#nn.Conv2d(1,5,kernel_size=3,stride=1,padding=1)#nn.Linear(6,30)
        self.bn1 = nn.BatchNorm1d(self.dim)
        self.layer1 = nn.Linear(self.dim,self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        self.layer3 = nn.Linear(self.dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, x):
        #out, _ = self.rnn(x)
        #out = F.normalize(out[-1,:,:])
        #out = F.dropout(out, p=0.5)
        x = torch.relu(self.layer1(x))
        x = F.dropout(x, p=0.5)
        #x = F.normalize(x)
        #x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        #x = F.relu(self.layer3(x))
        return x

class Value(nn.Module):
    def __init__(self, n_hotel, n_dyn):
        super(Value, self).__init__()
        self.dim = n_hotel+365
        self.rnn = nn.LSTM(self.dim,self.dim,num_layers=2, dropout=0.5)#nn.Conv2d(1,5,kernel_size=3,stride=1,padding=1)#nn.Linear(6,30)
        #self.bn1 = nn.BatchNorm1d(self.dim)
        self.layer1 = nn.Linear(self.dim+n_dyn,self.dim) # self.dim for state; n_hotel for action
        self.layer2 = nn.Linear(self.dim,self.dim)
        self.layer3 = nn.Linear(self.dim,1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, state, action):
        #out, _ = self.rnn(state)
        #out = F.normalize(out[-1,:,:])
        x = torch.relu(self.layer1(torch.cat((state,action),dim=1)))
        x = F.dropout(x,p=0.5)
        x = torch.tanh(self.layer2(x))
        x = F.dropout(x, p=0.5)
        #x = F.normalize(x)
        x = self.layer3(x)
        #x = F.relu(self.bn2(self.layer2(x))).view(x.size(0),-1)
        #x = F.relu(self.layer3(x))
        return x

class Agent():
    def __init__(self, dynamic_hotel, device, n_hotel, p_lr, p_l2, v_lr, v_l2,
                 memory_size, batch_size, eps_start, eps_end, eps_decay, gamma, tau, double_q, memory):
        super(Agent, self).__init__()
        self.dynamic_hotel = dynamic_hotel
        #self.dynamic_mask = np.array([0.0 if i not in self.dynamic_hotel else 1.0 for i in range(n_hotel)])
        self.n_hotel = n_hotel
        self.device = device
        #self.torch_mask = torch.tensor(self.dynamic_mask, dtype=torch.float).to(self.device)
        self.policy_eval_net = Policy(self.n_hotel, len(self.dynamic_hotel)).float().to(device)
        self.policy_target_net = Policy(self.n_hotel, len(self.dynamic_hotel)).float().to(device)

        self.value_eval_net = Value(self.n_hotel, 1).float().to(device) #len(self.dynamic_hotel_list)
        self.value_target_net = Value(self.n_hotel, 1).float().to(device)

        self.memory = memory
        self.start_learning = 10000
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.global_step = 0
        self.year_num = 365

        self.double_q = double_q
        self.policy_optimizer = optim.Adam(self.policy_eval_net.parameters(),
                                           lr=p_lr, weight_decay=p_l2)
        self.value_optimizer = optim.Adam(self.value_eval_net.parameters(),
                                          lr=v_lr, weight_decay=v_l2)

        self.loss_func = nn.MSELoss()
        self.policy_target_net.load_state_dict(self.policy_eval_net.state_dict())

    def choose_action(self, state, step, is_test = False):
        state = torch.tensor(state, dtype=torch.float).view(1, self.n_hotel+self.year_num).to(self.device)
        #obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.global_step * self.eps_decay)
        if is_test or random.random() > eps_threshold:
            action_value = eps_threshold*self.policy_eval_net(state).view(1, len(self.dynamic_hotel)).cpu().detach().numpy()
        else:
            action_value = (torch.rand([1, len(self.dynamic_hotel)]).numpy() - 0.5) * 2
        eval_val = 0#self.value_eval_net(state, (torch.tensor(action_value).to(self.device)))

        return action_value, eval_val

    def learn(self):
        #print(self.device)
        #if len(self.memory) < self.start_learning:
        #    return
        #dynamic_mask = torch.tensor(self.dynamic_mask, dtype=torch.float).to(self.device)
        #for target_param, param in zip(self.policy_target_net.parameters(), self.policy_eval_net.parameters()):
        #    target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

        self.global_step += 1

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = batch.state
        #self_state_batch = batch.self_state
        next_state_batch = batch.next_state
        action_batch = batch.action
        reward_batch = batch.reward

        #policy_loss = torch.tensor(0.0, dtype=torch.float).to(self.device)
        #value_loss = torch.tensor(0.0, dtype=torch.float).to(self.device)
        #abs_value_loss = torch.tensor(0.0, dtype=torch.float).to(self.device)
        #reward_diff = torch.tensor([[0.0]], dtype=torch.float).to(self.device)
        #mean_reward = torch.tensor([[0.0]], dtype=torch.float).to(self.device)
        #for b in range(self.batch_size):        step = state_batch.size()[1]
        cur_state = FloatTensor(torch.cat(state_batch).float().reshape(self.batch_size,self.n_hotel+self.year_num)).to(self.device)
        #self_state = FloatTensor(torch.cat(self_state_batch).float().reshape(self.batch_size,2)).to(self.device)
        next_state_ser = FloatTensor(torch.cat(next_state_batch).float().reshape(self.batch_size,self.n_hotel+self.year_num)).to(self.device)
        action = FloatTensor(torch.cat(action_batch).reshape(self.batch_size, -1).float()).to(self.device)
        reward = FloatTensor(torch.cat(reward_batch).reshape(self.batch_size, -1).float()).to(self.device)
        reward = F.normalize(reward, dim=0)

        policy_loss = -self.value_eval_net(cur_state, self.policy_eval_net(cur_state)).mean()

        next_action = self.policy_target_net(next_state_ser)
        target_value = self.value_target_net(next_state_ser, next_action.detach())

        exp_value = (reward[:,self.dynamic_hotel].reshape(self.batch_size,1) + self.gamma*target_value)

        mean_reward = torch.abs(exp_value.detach()).sum()/self.batch_size
        value = self.value_eval_net(cur_state, action[:,self.dynamic_hotel])
        value_loss = self.loss_func(value, exp_value.detach()) # torch.abs(value - exp_value.detach())#
        abs_value_loss = torch.abs(value).sum()/self.batch_size
        reward_diff = torch.abs(value - exp_value.detach()).sum()/(self.batch_size)

        value_loss = value_loss/mean_reward
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        #print()
        #print("Policy Loss: " + str(policy_loss.cpu().detach().numpy()))
        #print("Value Loss: " + str(value_loss.cpu().detach().numpy()))
        #print("Reward mean diff: " + str(reward_diff.cpu().detach().numpy()/abs_value_loss.cpu().detach().numpy()))
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        #max_grad_norm = 0.5
        #torch.nn.utils.clip_grad_norm_(self.value_eval_net.parameters(), max_grad_norm)
        #torch.nn.utils.clip_grad_norm_(self.policy_eval_net.parameters(), max_grad_norm)

        for target_param, param in zip(self.value_target_net.parameters(), self.value_eval_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.policy_target_net.parameters(), self.policy_eval_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        return policy_loss.cpu().detach().numpy(), \
               value_loss.cpu().detach().numpy(), \
               reward_diff.cpu().detach().numpy()/abs_value_loss.cpu().detach().numpy()





