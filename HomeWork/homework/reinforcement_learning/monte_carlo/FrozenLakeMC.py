import sys
import gym
import numpy as np
from collections import defaultdict


class FrozenLakeMC:

    def __init__(self):
        self.env = gym.make('FrozenLake-v1', is_slippery=False)

    def print_environment_info(self):
        print('Observations: {}'.format(self.env.observation_space))
        print('Actions: {}'.format(self.env.action_space))
