import gym
from gym import spaces
import numpy as np
import torch

from dataset import MyDataset
from sharpe import Sharpe
from utils import EnvConfig
import pandas as pd


class MyEnv(gym.Env):

    def __init__(self, df_train, df_test, play=False):
        self.df_train = df_train
        self.df_test = df_test
        self.play = play
        if not self.play:
            self.df = self.df_train
        else:
            self.df = self.df_test
        self.config = EnvConfig()
        self.current_idx = 0

        self.weights = np.full((len(self.config.stocks)), 1 / len(self.config.stocks), dtype=float)

        # cash per stock_values
        self.cash = self.weights * self.config.initial_cash

        self.portfolio_value = 0
        self.stock_values = np.zeros(len(self.config.stocks))
        self.provisions = np.zeros(len(self.config.stocks))

        self.states = self.df.loc[:, ~self.df.columns.isin(self.config.stocks_adj_close_names)].to_numpy()
        self.rewards = self.df[self.config.stocks].to_numpy()

        self.n = len(self.states)

        self.low = -100
        self.high = 100
        ######
        # self.discretized = np.linspace(self.low, self.high, self.bins).astype(int)
        increment = 1
        self.discretized = np.arange(start=self.low, stop=self.high + increment, step=increment)

        self.action_space = spaces.Box(
            low=self.low,
            high=self.high, shape=(len(self.stock_values),),
            dtype=int
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.nan,
            shape=(self.states.shape[1],),  # same as number of stocks
            dtype=np.float32
        )

    def reset(self):
        self.weights = np.full((len(self.config.stocks)), 1 / len(self.config.stocks), dtype=float)
        self.cash = self.weights * self.config.initial_cash
        self.portfolio_value = 0
        self.provisions = np.zeros(len(self.config.stocks))
        self.stock_values = np.zeros(len(self.config.stocks))

        self.current_idx = 0

        next_state = self.states[self.current_idx]
        next_state = np.array(next_state).reshape(1, -1)
        next_state = torch.tensor(next_state).float().to(self.config.device)
        return next_state

    def get_sharpe(self, play, env_action_index):
        if env_action_index is None:
            stock_names = self.config.stocks
        else:
            stock_names = np.array(self.config.stocks)[env_action_index]
        if not play:
            # if self.current_idx % 30 == 0 and self.current_idx != 0:
            log_returns = self.df.iloc[:self.current_idx + 1][stock_names].tail(60)
        else:
            df_data = pd.concat([self.df_train, self.df_test])
            log_returns = df_data.iloc[:len(self.df_train) + self.current_idx + 1][stock_names].tail(60)

        if len(log_returns.columns) != 0:
            sharpe = Sharpe(log_returns)
        else:
            sharpe = None
        return sharpe

    def step(self, actions):

        if self.current_idx >= self.n:
            raise Exception("Episode already done")

        actions = np.array([self.discretized[aa] for aa in actions])

        # actions is an array and contains one action per stock_values
        actions = actions.ravel()

        sharpe = self.get_sharpe(self.play, None)
        self.weights = sharpe.get_weights()
        # ret, vol, sh = sharpe.get_ret_vol_sr(self.weights)
        current_prices = self.df.iloc[self.current_idx][self.config.stocks_adj_close_names].values
        # current_price = np.dot(self.weights, current_prices) - # todo Fehler?

        self.cash = self.weights * np.sum(self.cash)
        env_actions = np.zeros(len(actions))
        provisions = 0.02
        for i, action in enumerate(actions):
            buy_max = self.cash[i] // (current_prices[i] * (1+provisions))  # with provisions
            #buy_max = self.cash[i] // current_prices[i]  # ohne provisions

            # buy or sell stocks
            if action > 0:
                # buy "action" stocks or max if "action" is higher
                env_actions[i] = np.minimum(buy_max, action)
            else:
                # env_action = - np.minimum(self.stock_values, np.abs(action * self.weights)) # todo
                # sell "action" stocks or all of "action" is higher
                env_actions[i] = - np.minimum(self.stock_values[i], np.abs(action))

            self.stock_values[i] += env_actions[i]

            # update cash
            if action > 0: # buy
                self.cash[i] -= env_actions[i] * current_prices[i] * (1 + provisions)
            else:  # sell
                self.cash[i] += - env_actions[i] * current_prices[i] * (1 + provisions)

            provision = np.abs(env_actions[i]) * current_prices[i] * provisions

            self.provisions[i] += provision

        # state transition
        done = (self.current_idx == self.n - 1)

        self.current_idx += 1

        if not done:

            if action is None:
                raise Exception("NaNs detected!")

            # compute reward
            # print("return", np.sum(env_action * current_price * (np.exp(self.rewards[self.current_idx]) - 1)))
            # print("provisions", np.sum(.01 * np.abs(env_action) * current_price))
            step_reward = np.sum(env_actions * current_prices * (np.exp(self.rewards[self.current_idx]) - 1)) - np.sum(.001 * np.abs(env_actions) * current_prices)
            next_prices = self.df.iloc[self.current_idx][self.config.stocks_adj_close_names].values
            self.portfolio_value = np.dot(self.stock_values, next_prices)

            invested_index = np.where(self.stock_values != 0) # invested
            
            invested_sharpe = self.get_sharpe(self.play, invested_index)
            if invested_sharpe is not None:
                invested_weights = invested_sharpe.get_weights()
                ret, vol, sh = invested_sharpe.get_ret_vol_sr(invested_weights)
            else:
                sh = -1.
                vol = 10
            
            reward = self.portfolio_value * sh + step_reward * 252

            #reward = self.portfolio_value



            next_state = self.states[self.current_idx]
            #kernel = self.states[self.current_idx:self.current_idx+7, 1][::-1]
            #print(kernel)
            next_state = np.array(next_state).reshape(1, -1)
            next_state = torch.tensor(next_state).reshape(1, -1).float().to(self.config.device)

        else:
            next_state = None
            reward = 0
            self.portfolio_value = np.dot(self.stock_values, current_prices)

        return next_state, reward, done


if __name__ == "__main__":
    dataset = MyDataset()
    train, test = dataset.get_train_test()

    train_env = MyEnv(train, test)
    test_env = MyEnv(train, test, play=True)
    num_states = train_env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = train_env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = train_env.action_space.high[0]
    lower_bound = train_env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    action = [np.random.randint(0, 99, 1)]
    print(action)

    first_state = test_env.reset()
    next_state, reward, done = train_env.step(action)
    print(first_state, next_state, reward, done)
    print(np.arange(start=-100, stop=100 + 1, step=1))
