from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import gym
import random

from model import DuelingNetwork, BranchingQNetwork
from utils import TensorEnv, ExperienceReplayMemory, AgentConfig, BranchingTensorEnv
import utils

from dataset import MyDataset
from env import MyEnv
from vae import VAE, Trainer

from sklearn.preprocessing import StandardScaler

import os

# from sdv.timeseries import PAR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BranchingDQN(nn.Module):

    def __init__(self, obs, ac, n, config):

        super().__init__()
        self.config = config
        self.q = BranchingQNetwork(obs, ac, n).to(self.config.device)
        self.target = BranchingQNetwork(obs, ac, n).to(self.config.device)
        self.target.load_state_dict(self.q.state_dict())

        self.target_net_update_freq = config.target_net_update_freq
        self.update_counter = 0

    def get_action(self, x):

        with torch.no_grad():
            # a = self.q(x).max(1)[1]
            out = self.q(x).squeeze(0)
            action = torch.argmax(out, dim=1)
        return action.cpu().numpy()

    def update_policy(self, adam, memory, params):

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params.batch_size)

        states = torch.tensor(b_states).float().to(self.config.device)
        actions = torch.tensor(b_actions).long().reshape(states.shape[0], -1, 1).to(self.config.device)
        rewards = torch.tensor(b_rewards).float().reshape(-1, 1).to(self.config.device)
        next_states = torch.tensor(b_next_states).float().to(self.config.device)
        masks = torch.tensor(b_masks).float().reshape(-1, 1).to(self.config.device)

        # qvals = self.q(states)

        current_q_values = self.q(states).gather(2, actions).squeeze(-1)

        with torch.no_grad():

            argmax = torch.argmax(self.q(next_states), dim=2)

            max_next_q_vals = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_vals = max_next_q_vals.mean(1, keepdim=True)

        expected_q_vals = rewards + max_next_q_vals * 0.99 * masks  # Belmann
        # print(expected_q_vals[:5])
        loss = F.mse_loss(expected_q_vals, current_q_values)

        # input(loss)

        # print('\n'*5)

        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-1., 1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            self.target.load_state_dict(self.q.state_dict())


def train():
    args = utils.arguments()
    config = AgentConfig()
    env_config = utils.EnvConfig()

    # torch.cuda.init()  # To test
    # print("torch.cuda.is_available(): ", torch.cuda.is_available())  # To test: returned true
    # print("torch.cuda.is_initialized(): ", torch.cuda.is_initialized())

    # env = BranchingTensorEnv(args.env, bins)

    dataset = MyDataset()
    train, test = dataset.get_train_test()
    env = MyEnv(train, test)

    if True: #dataset.outdated_data_files:
        trainer = Trainer()

        for epoch in range(1, trainer.vae_config.epochs + 1):
            trainer.train(epoch)
            trainer.test(epoch)
            with torch.no_grad():
                sample = torch.randn(64, 20).to(trainer.device)
                sample = trainer.model.decode(sample).cpu()

            trainer.save()

    vae = VAE(train.shape[1] - len(env_config.stocks_adj_close_names)).to(config.device)
    vae.load_state_dict(torch.load('./runs/vae/vae_state_dict'))

    """with torch.no_grad():
        data = torch.tensor(train.values).float().to(device)
        recons, _, _ = vae(data)


    train_gen = pd.DataFrame(np.array(recons), columns=train.columns)
    env_vae = MyEnv(train_gen)"""

    # env_val = MyEnv(val)

    memory = ExperienceReplayMemory(config.memory_size)
    agent = BranchingDQN(env.observation_space.shape[0], env.action_space.shape[0], config.bins, config)
    adam = optim.Adam(agent.q.parameters(), lr=config.lr)

    s = env.reset()
    ep_reward = 0.
    recap = []
    episode = 0
    portfolio_hist = []
    cash_hist = []
    stock_hist = []
    action_frame = []

    p_bar = tqdm(total=config.max_frames)
    for frame in range(config.max_frames):

        epsilon = config.epsilon_by_frame(frame)

        if np.random.random() > epsilon:
            action = agent.get_action(s)
        else:
            action = np.random.randint(0, config.bins, size=env.action_space.shape[0])

        action_frame.append(action)
        ns, r, done = env.step(action)

        if frame % 4 == 0:
            if ns is not None:
                with torch.no_grad():
                    sample, _, _ = vae(ns)
                    # sample = torch.randn(1, 20).to(device)
                    # sample = vae.decode(sample).cpu()
                ns = sample

        ep_reward += r

        if done:
            episode += 1
            print('Finished episode', episode)
            recap.append(ep_reward)
            portfolio_hist.append(env.portfolio_value)
            cash_hist.append(np.sum(env.cash))  # todo separate cash by stock
            stock_hist.append(env.stock_values)

            # p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
            p_bar.set_description('Portfolio: {:.3f}'.format(env.portfolio_value) +
                                  ' Cash: {:.3f}'.format(np.sum(env.cash)) +
                                  " ".join([' ' + stock + ': {:.3f}'.format(stock_value) for stock, stock_value in
                                            zip(env_config.stocks, env.stock_values) if int(stock_value) != 0]))
            ep_reward = 0.
            ns = env.reset()

        memory.push((s.cpu().reshape(-1).numpy().tolist(), action, r, ns.cpu().reshape(-1).numpy().tolist(),
                     0. if done else 1.))

        s = ns

        p_bar.update(1)

        if frame > config.learning_starts:
            agent.update_policy(adam, memory, config)

        if frame % 1000 == 0 and frame != 0:
            utils.save(agent, (recap, portfolio_hist, cash_hist, stock_hist), args)
            # play auf valid

        if episode == config.max_episodes:
            break

    p_bar.close()


if __name__ == "__main__":
    train()
