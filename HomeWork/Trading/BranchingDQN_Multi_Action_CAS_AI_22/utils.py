import datetime

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


def arguments():
    parser = ArgumentParser()
    parser.add_argument('--env', default='House_Of_Money')

    return parser.parse_args()


def save(agent, hist, args):
    config = EnvConfig()
    stocks = config.stocks
    now = datetime.datetime.now()
    now = now.strftime("%d.%m.%Y")
    path = './runs/{}/{}/'.format(args.env, now)
    try:
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict'))
    except:
        pass

    rewards, portfolio_hist, cash_hist, stock_hist = hist
    if len(stock_hist) > 0:
        stock_hist = np.vstack(stock_hist)
        stocks = pd.DataFrame(stock_hist, columns=stocks)
        stocks.plot()

        plt.xlabel('Episodes')
        plt.ylabel('Stocks')
        # plt.title('Branching DDQN: {}'.format(args.env))
        plt.title('Stocks')

        plt.savefig(os.path.join(path, 'stocks.png'))

    plt.cla()
    plt.clf()
    plt.plot(rewards, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(rewards, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    # plt.title('Branching DDQN: {}'.format(args.env))
    plt.title('Cumulative reward')
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index=False)

    plt.cla()
    plt.clf()
    plt.plot(portfolio_hist, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(portfolio_hist, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Value of Portfolio')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'portfolio.png'))

    plt.cla()
    plt.clf()
    plt.plot(cash_hist, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(cash_hist, sigma=5), c='r', label='Cash')
    plt.xlabel('Episodes')
    plt.ylabel('$Cash$')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'cash.png'))

    plt.cla()
    plt.clf()
    plt.plot(np.array(portfolio_hist) + np.array(cash_hist), c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(np.array(portfolio_hist) + np.array(cash_hist), sigma=5), c='r',
             label='Cash + Portfolio')
    plt.xlabel('Episodes')
    plt.ylabel('$Cash$ + Portfolio')
    plt.title('Cash + Portfolio')
    plt.savefig(os.path.join(path, 'cash_and_portfolio.png'))


class EnvConfig:

    def __init__(self,  # more stocks ['AAPL', 'MSFT', 'AMZN']
                 stocks=['JPM', 'T', 'AAPL', 'AMZN'],
                 #  'XLC' removed since it starts in 2018
                 # stocks=['XLE', 'XLP', 'XLI', 'XLF', 'XLK',  'XLU', 'XLV', 'XLY', 'XLB'],
                 initial_cash=2_000):
        self.stocks = stocks
        self.initial_cash = initial_cash
        self.stocks_adj_close_names = [stock + '_Adj_Close' for stock in self.stocks]
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda = True
        cuda = self.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")


class AgentConfig:

    def __init__(self,
                 epsilon_start=1.,
                 epsilon_final=0.01,
                 epsilon_decay=10 * 8000,
                 gamma=0.99,
                 lr=1e-4,
                 target_net_update_freq=1000,
                 memory_size=100000,
                 batch_size=128,
                 learning_starts=5000,
                 max_frames=10_000_000,
                 max_episodes=60,
                 bins=201,
                 tau=0.1):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_by_frame = lambda i: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -1. * i / self.epsilon_decay)

        self.gamma = gamma
        self.lr = lr

        self.tau = tau

        self.target_net_update_freq = target_net_update_freq
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.learning_starts = learning_starts
        self.max_frames = max_frames
        self.max_episodes = max_episodes
        self.bins = bins
        self.cuda = True
        cuda = self.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"


class VAEConfig:

    def __init__(self,
                 batch_size=64,
                 epochs=30,
                 cuda=False,
                 seed=1,
                 log_interval=10,
                 save_interval=10):
        self.batch_size = batch_size
        self.epochs = epochs
        self.cuda = cuda
        self.seed = seed
        self.save_interval = save_interval
        self.log_interval = log_interval


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
        return torch.tensor(x).reshape(1, -1).float()

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
        self.discretized = [np.linspace(self.action_space.low[i], self.action_space.high[i], self.n) for i in
                            range(len(self.action_space.low))]

    def step(self, a):
        actions = []
        for i, a in enumerate(a):
            actions.append(self.discretized[i][a])
        return super().step(np.array(actions))
