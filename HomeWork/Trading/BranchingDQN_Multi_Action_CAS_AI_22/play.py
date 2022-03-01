from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import gym
import random
import time

from branching_dqn import BranchingDQN
from dataset import MyDataset
from env import MyEnv
from model import BranchingQNetwork
from utils import TensorEnv, ExperienceReplayMemory, AgentConfig, BranchingTensorEnv, EnvConfig
import utils
import datetime


def run():
    args = utils.arguments()
    now = datetime.datetime.now()
    now = now.strftime("%d.%m.%Y")
    path = './runs/{}/{}/model_state_dict'.format(args.env, now)
    print("Loading the agent model. Please, control the date:", path)

    # env = BranchingTensorEnv(args.env, bins)

    dataset = MyDataset()

    train, test = dataset.get_train_test()

    env = MyEnv(train, test, play=True)
    agent_config = AgentConfig()
    env_config = EnvConfig()

    agent = BranchingDQN(env.observation_space.shape[0], env.action_space.shape[0], agent_config.bins, agent_config)
    agent.q.load_state_dict(torch.load(path))

    s = env.reset()
    done = False
    ep_reward = 0

    pbar = tqdm(total=len(test))
    while not done:
        with torch.no_grad():
            out = agent.q(s).squeeze(0)
        action = torch.argmax(out, dim=1).cpu().numpy().reshape(-1)
        s, r, done = env.step(action)
        ep_reward += r
        pbar.update(1)
    pbar.close()

    print('Ep reward: {:.3f}'.format(ep_reward))
    print('Portfolio: {:.3f}'.format(env.portfolio_value))
    print('Cash: {:.3f}'.format(np.sum(env.cash)))


    print('Provision: {:.3f}'.format(np.sum(env.provisions)))

    env.close()


if __name__ == "__main__":
    run()
