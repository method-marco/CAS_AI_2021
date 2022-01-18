import gym
import numpy as np


class TradingEnv(gym.Env):

    def __init__(self, df_train, df_test, play=False, initial_cash=10000):
        self.df_train = df_train
        self.df_test = df_test
        self.play = play

        if not self.play:
            df_data = self.df_train
        else:
            df_data = self.df_test

        self.current_idx = 0
        self.stocks = ['AAPL', 'MSFT', 'NFLX', 'AMZN'] # target
        self.stocks_adj_close_names = [stock + '_Adj_Close' for stock in self.stocks] # Name of columns df

        self.initial_cash = initial_cash
        self.cash = 0
        self.portfolio_value = 0
        self.stock_values = np.zeros(len(self.stocks))

        self.states = df_data.loc[:, ~df_data.columns.isin(self.stocks_adj_close_names)].to_numpy()

    def reset(self):
        self.current_idx = 0

        self.cash = self.initial_cash
        self.portfolio_value = 0
        self.stock_values = np.zeros(len(self.stocks))

        next_state = self.states[self.current_idx]
        return next_state

    def step(self, actions):
        pass


