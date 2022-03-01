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

from model import BranchingQNetwork 
import utils

if __name__ == '__main__':
    env_name, config = utils.get_luna_lander_settings()
    env = utils.BranchingTensorEnv(env_name, config.bins)

    agent = BranchingQNetwork(env.observation_space.shape[0], env.action_space.shape[0], config.bins)
    agent.load_state_dict(torch.load('./runs/{}/model_state_dict'.format(env_name)))

    print(agent)
    for ep in tqdm(range(10)):

        s = env.reset()
        done = False
        ep_reward = 0
        while not done:

            with torch.no_grad():
                out = agent(s).squeeze(0)
            action = torch.argmax(out, dim = 1).numpy().reshape(-1)
            #print(action)
            s, r, done, _ = env.step(action)

            env.render()
            ep_reward += r

        print('\nEp reward: {:.3f}'.format(ep_reward))

    env.close()