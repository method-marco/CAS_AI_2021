import numpy as np 
import gym 
import torch 
import random
from argparse import ArgumentParser 
import os 
import pandas as pd 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d

def get_luna_lander_settings():
    env_name = 'LunarLanderContinuous-v2'

    config = AgentConfig()
    config.epsilon_start = 1.
    config.epsilon_final = 0.01
    config.epsilon_decay = 500000
    config.gamma = 0.99
    config.lr = 1e-5
    config.target_net_update_freq = 1000
    config.memory_size = 100000
    config.batch_size = 128
    config.learning_starts = 100000
    config.max_frames = 5000000
    config.tau = 0.2
    config.bins = 60

    return env_name, config

def get_bipedal_walker_settings():
    env_name = 'BipedalWalker-v3'

    config = AgentConfig()
    config.epsilon_start = 1.
    config.epsilon_final = 0.01
    config.epsilon_decay = 8000
    config.gamma = 0.99
    config.lr = 1e-4
    config.target_net_update_freq = 1000
    config.memory_size = 100000
    config.batch_size = 128
    config.learning_starts = 5000
    config.max_frames = 10000000
    config.tau = 0.1
    config.bins = 6

    return env_name, config


def save(agent, rewards, env_name):

    path = './runs/{}/'.format(env_name)
    try: 
        os.makedirs(path)
    except: 
        pass 

    torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict'))

    plt.cla()
    plt.plot(rewards, c = 'r', alpha = 0.3)
    plt.plot(gaussian_filter1d(rewards, sigma = 5), c = 'r', label = 'Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Branching DDQN: {}'.format(env_name))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)





class AgentConfig:

    def __init__(self, 
                 epsilon_start = 1.,
                 epsilon_final = 0.01,
                 epsilon_decay = 8000,
                 gamma = 0.99, 
                 lr = 1e-4, 
                 target_net_update_freq = 1000, 
                 memory_size = 100000, 
                 batch_size = 128, 
                 learning_starts = 5000,
                 max_frames = 10000000,
                 tau = 0.1):

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_by_frame = lambda i: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * i / self.epsilon_decay)

        self.gamma =gamma
        self.lr =lr

        self.target_net_update_freq =target_net_update_freq
        self.memory_size =memory_size
        self.batch_size =batch_size

        self.learning_starts = learning_starts
        self.max_frames = max_frames
        self.tau = tau


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])


        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class TensorEnv(gym.Wrapper): 

    def __init__(self, env_name): 

        super().__init__(gym.make(env_name))

    def process(self, x): 

        return torch.tensor(x).reshape(1,-1).float()

    def reset(self): 

        return self.process(super().reset())

    def step(self, a): 

        ns, r, done, infos = super().step(a)
        return self.process(ns), r, done, infos 


class BranchingTensorEnv(TensorEnv): 

    def __init__(self, env_name, n):

        super().__init__(env_name)
        self.n = n
        # take from env
        self.discretized = [np.linspace(self.action_space.low[i], self.action_space.high[i], self.n) for i in range(len(self.action_space.low))]


    def step(self, a):
        actions = []
        for i, a in enumerate(a):
            actions.append(self.discretized[i][a])
        return super().step(np.array(actions))
