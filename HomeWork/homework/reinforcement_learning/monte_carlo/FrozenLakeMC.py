import sys
import gym
import numpy as np
from collections import defaultdict


class FrozenLakeMC:

    def __init__(self):
        env = gym.make('FrozenLake-v0', is_slippery=False)
