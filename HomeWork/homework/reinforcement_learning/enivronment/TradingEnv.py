import gym
import numpy as np


class TradingActions:
    Hold = 0
    Buy = 1
    Sell = 2


class TradingEnv(gym.Env):

    def __init__(self, stocks, df_train, df_test, play=False, initial_cash=10000):
        self.df_train = df_train
        self.df_test = df_test
        self.play = play

        if not self.play:
            self.df_data = self.df_train
        else:
            self.df_data = self.df_test

        self.current_idx = 0
        self.stocks = stocks
        self.stocks_adj_close_names = [stock + '_Adj_Close' for stock in self.stocks]

        self.initial_cash = initial_cash
        self.cash = 0
        self.portfolio_value = 0
        self.stock_values = np.zeros(len(self.stocks))

        # remove stocks
        self.states = self.df_data.loc[:, ~self.df_data.columns.isin(self.stocks_adj_close_names)].to_numpy()
        self.n_states = len(self.states)

    def reset(self):
        self.current_idx = 0

        self.cash = self.initial_cash
        self.portfolio_value = 0
        self.stock_values = np.zeros(len(self.stocks))

        next_state = self.states[self.current_idx]
        return next_state

    def step(self, action):
        reward = 0

        if self.current_idx >= self.n_states:
            raise Exception('Episode already done')

        # state transistion
        done = (self.current_idx == self.n_states - 1)

        # TODO: apply action
        self.current_idx += 1
        if not done:
            # TODO: compute reward
            stock_increase = self.df_data.iloc[self.current_idx]['AAPL_Adj_Close'] > 0

            if action == TradingActions.Hold and stock_increase:
                reward = 0
            elif action == TradingActions.Hold and not stock_increase:
                reward = -1
            elif action == TradingActions.Buy and stock_increase:
                reward = 1
            elif action == TradingActions.Buy and not stock_increase:
                reward = -1
            elif action == TradingActions.Sell and stock_increase:
                reward = -1
            elif action == TradingActions.Sell and not stock_increase:
                reward = 1

            next_state = self.states[self.current_idx]
        else:
            # done
            next_state = None
            reward = 0

        return next_state, reward, done
