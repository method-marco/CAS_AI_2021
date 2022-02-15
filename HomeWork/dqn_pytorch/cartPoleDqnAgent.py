import collections
import os
import random
from typing import Deque
import numpy as np
import torch

import gym
from cartPoleDqn import DQN


PROJECT_PATH = os.path.abspath("/Users/mr_marco/Documents/source/CAS_AI_2021/HomeWork/dqn_pytorch")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "dqn_cartpole.h5")
TARGET_MODEL_PATH = os.path.join(MODELS_PATH, "target_dqn_cartpole.h5")


class Agent:
    def __init__(self, env: gym.Env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 10_000
        self.train_start = 1_000
        self.memory: Deque = collections.deque(
            maxlen=self.replay_buffer_size
        )
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3

        self.dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate
        )
        self.target_dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate
        )
        self.target_dqn.update_model(self.dqn)
        self.target_dqn.eval()
        self.batch_size = 32

    def get_action(self, state: torch.Tensor):
        if torch.rand(1).item() <= self.epsilon:
            return torch.randint(self.actions, (1,)).item()
        else:
            return torch.argmax(self.dqn(state)).item()

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=5)
        best_reward_mean = 0.0

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = torch.from_numpy(self.env.reset())
            state = torch.reshape(state, (1, -1)).float()

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.reshape(torch.from_numpy(next_state), (1, -1)).float()
                if done and total_reward < 499:
                    reward = -100.0
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state

                if done:
                    if total_reward < 500:
                        total_reward += 100.0
                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)

                    if current_reward_mean > best_reward_mean:
                        self.target_dqn.update_model(self.dqn)
                        best_reward_mean = current_reward_mean
                        self.dqn.save_model(MODEL_PATH)
                        self.target_dqn.save_model(TARGET_MODEL_PATH)
                        print(f"New best mean: {best_reward_mean}")

                        if best_reward_mean > 400:
                            return
                    break

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = torch.cat(states).type(torch.float32)
        states_next = torch.cat(states_next).type(torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        q_values = self.dqn(states)
        q_values_next = self.target_dqn(states_next).detach()
        q_values_expected = q_values.detach().clone()

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values_expected[i][a] = rewards[i]
            else:
                q_values_expected[i][a] = rewards[i] + self.gamma * torch.max(q_values_next[i])

        self.dqn.fit(q_values, q_values_expected)

    def play(self, num_episodes: int, render: bool = True):
        self.dqn.load_model(MODEL_PATH)
        self.target_dqn.load_model(TARGET_MODEL_PATH)

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = torch.from_numpy(state)
            state = torch.reshape(state, (1, -1)).float()

            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state)
                next_state = torch.reshape(next_state, (1, -1)).float()
                total_reward += reward
                state = next_state

                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=250)
    input("Play?")
    agent.play(num_episodes=20, render=True)
