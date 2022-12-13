# -*- coding: utf-8 -*-
"""
    REFERENCE FOR THE CODE - 
    https://github.com/antocapp/paperspace-ddpg-tutorial/blob/master/ddpg-pendulum-250.ipynb

    Install OpenAI Gym version == 0.7.4
"""

import gym
import numpy as np
import random
from gym import Env
from gym.spaces import Box
from gym.error import DependencyNotInstalled
import torch
from torch import nn #needed for building neural networks
import torch.nn.functional as F #needed for activation functions
import torch.optim as opt #needed for optimisation
from tqdm import tqdm_notebook as tqdm
from copy import copy, deepcopy
from collections import deque
from matplotlib import pyplot as plt
from IPython.display import clear_output
print("Using torch version: {}".format(torch.__version__))
print("Using gym version: {}".format(gym.__version__))

class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take
        self.max_knob = np.pi
        # temperatures of the shower
        self.max_temperature = 100
        # action space
        self.action_space = Box(low=-self.max_knob, high=self.max_knob, shape=(1,))
        # observation_space
        self.observation_space = Box(low=np.array([0]), high=np.array([self.max_temperature]))
        # initial state
        self.state = 38 + np.random.randint(-3, 3)
        # episode length
        self.shower_length = 60
        self.ratio = 0.25
        # self.max_reward = 10

        self.a = -1
        self.b = -1

    def step(self, action):
        if action < 0 and action >= -self.max_knob:
            self.state[0] -= 1
        if action > 0 and action <= self.max_knob:
            self.state[0] += 1
        self.shower_length -= 1

        if self.state[0] >= 37 and self.state[0] <= 39:
            reward = 10
        elif self.state[0] > 39:
            i = self.state[0] - 38
            reward = 10 - 2*i[0]
        else:
            i = self.state[0] - 38
            reward = 10 + 2*i[0]

        # check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Apply temperature noise
        self.state[0] += np.random.randint(-1, 1)

        # state placeholder for info
        info = {}

        # return step information
        return self.state, reward, done, info

    def render(self):
        # visulization Implementation
        pass
    def reset(self):
        self.state = np.array([38 + np.random.randint(-3, 3)]).reshape((1,1))
        self.shower_length = 60
        return self.state

env = ShowerEnv()

buffer_size=1000000
size_batch=64
gamma=0.99
tau=0.001       # HyperParameters Update rate for Target Networks
lr_actor=0.0001      # Learning rate for actor
lr_critic=0.001       # Learning rate for actor
H1=400   # Neurons of 1st hidden layer
H2=300   # Neurons of 2nd hidden layer

max_episodes=2000 # Mumber of episodes of the training
max_steps=200    # Step size of each episode. 
buffer_start = 100 # Initial warmup before training
epsilon = 1
epsilon_decay = 1./100000 

print_every = 10 #Print info about average reward every print_every

class replayBuffer(object):
    def __init__(self, buffer_size, name_buffer=''):
        self.buffer_size=buffer_size  #choose buffer size
        self.num_exp=0
        self.buffer=deque()

    def add(self, s, a, r, t, s2):
        experience=(s, a, r, t, s2)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, size_batch):
        if self.num_exp < size_batch:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, size_batch)

        s, a, r, t, s2 = map(np.stack, zip(*batch))

        return s, a, r, t, s2

    def clear(self):
        self.buffer = deque()
        self.num_exp=0

#set GPU for faster training
cuda = torch.cuda.is_available() #check for CUDA
device   = torch.device("cuda" if cuda else "cpu")
print("Job will run on {}".format(device))

def fanin_(size):
    """
        WEIGHT COMPUTATION FOR ACTOR AND CRITIC
    """
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

class Critic(nn.Module):
    """
        CRITIC NETWORK 
    """
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=3e-3):
        super(Critic, self).__init__()
                
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        
        self.linear2 = nn.Linear(h1+action_dim, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
                
        self.linear3 = nn.Linear(h2, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(torch.cat([x,action],2))
        x = self.relu(x)
        x = self.linear3(x)
        
        return x

class Actor(nn.Module): 
    """
        ACTOR NETWORK
    """
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=0.003):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        
        self.linear2 = nn.Linear(h1, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        
        self.linear3 = nn.Linear(h2, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x
    
    def get_action(self, state):
        # Actor calculates the action for a given state
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]

class Exploration:
    """
        ADDING ORNSTEIN-UHLENBECK PROCESS FOR RANDOMIZATION OF THE STATE SPACE
    """
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'Exploration(mu={}, sigma={})'.format(self.mu, self.sigma)

class NormalizedEnv(gym.ActionWrapper):
    """ 
        WRAP ACTION 
    """
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

env.reset()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("State dim: {}, Action dim: {}".format(state_dim, action_dim))

noise = Exploration(mu=np.zeros(action_dim))

critic  = Critic(state_dim, action_dim).to(device)
actor = Actor(state_dim, action_dim).to(device)

# Target Networks
target_critic  = Critic(state_dim, action_dim).to(device)
target_actor = Actor(state_dim, action_dim).to(device)

for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_actor.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
    
# Adam Optimizer for both actor and critic networks
q_optimizer  = opt.Adam(critic.parameters(),  lr=lr_critic) #, weight_decay=0.01)
policy_optimizer = opt.Adam(actor.parameters(), lr=lr_actor)

MSE = nn.MSELoss()

memory = replayBuffer(buffer_size)

def subplot(R, P, Q, S):
    r = list(zip(*R))
    p = list(zip(*P))
    q = list(zip(*Q))
    s = list(zip(*S))
    clear_output(wait=True)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'r') #row=0, col=0
    ax[1, 0].plot(list(p[1]), list(p[0]), 'b') #row=1, col=0
    ax[0, 1].plot(list(q[1]), list(q[0]), 'g') #row=0, col=1
    ax[1, 1].plot(list(s[1]), list(s[0]), 'k') #row=1, col=1
    ax[0, 0].title.set_text('Reward')
    ax[1, 0].title.set_text('Policy loss')
    ax[0, 1].title.set_text('Q loss')
    ax[1, 1].title.set_text('Max steps')
    plt.show()

episodes = 10
for episode in range(0, episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print(f'episode: {episode}, score: {score}')

plot_reward = []
plot_policy = []
plot_q = []
plot_steps = []


best_reward = -np.inf
saved_reward = -np.inf
saved_ep = 0
average_reward = 0
global_step = 0

for episode in range(max_episodes):
    s = env.reset()
    done = False
    ep_reward = 0.
    ep_q_value = 0.
    step=0

    for step in range(max_steps):
        global_step +=1
        epsilon -= epsilon_decay
        a = actor.get_action(s)

        a += noise()*max(0, epsilon)
        a = np.clip(a, -1., 1.)
        s2, reward, terminal, info = env.step(a)

        memory.add(s, a, reward, terminal,s2)

        if memory.count() > buffer_start:
            s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(size_batch)

            s_batch = torch.FloatTensor(s_batch).to(device)
            a_batch = torch.FloatTensor(a_batch).to(device)
            r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
            t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
            s2_batch = torch.FloatTensor(s2_batch).to(device)
            
            
            #compute loss for critic
            a2_batch = target_actor(s2_batch)
            a = torch.cat([s2_batch, a2_batch])
            target_q = target_critic(s2_batch, a2_batch) 
            y = r_batch + (1.0 - t_batch) * gamma * target_q.detach()
            q = critic(s_batch, a_batch)
            
            q_optimizer.zero_grad()
            q_loss = MSE(q, y) 
            q_loss.backward()
            q_optimizer.step()
            
            #compute loss for actor
            policy_optimizer.zero_grad()
            policy_loss = -critic(s_batch, actor(s_batch))
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            policy_optimizer.step()

            #soft update of the frozen target networks
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

        s = deepcopy(s2)
        ep_reward += reward

    try:
        plot_reward.append([ep_reward, episode+1])
        plot_policy.append([policy_loss.cpu().data, episode+1])
        plot_q.append([q_loss.cpu().data, episode+1])
        plot_steps.append([step+1, episode+1])
    except:
        continue
    average_reward += ep_reward
    
    if ep_reward > best_reward:
        torch.save(actor.state_dict(), 'best_model.pkl') 
        best_reward = ep_reward
        saved_reward = ep_reward
        saved_ep = episode+1

    if (episode % print_every) == (print_every-1):  
        subplot(plot_reward, plot_policy, plot_q, plot_steps)
        print('[%6d episode, %8d total steps] average reward for past {} iterations: %.3f'.format(print_every) %
            (episode + 1, global_step, average_reward / print_every))
        print("Last model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))
        average_reward = 0 #reset average reward

