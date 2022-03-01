from scipy.optimize import minimize
import numpy as np

from dataset import MyDataset
from utils import EnvConfig


class Sharpe:

    def __init__(self, log_ret):
        # By convention of minimize function it should be a function that returns zero for conditions
        self.cons = ({'type': 'eq', 'fun': self.check_sum})
        self.log_ret = log_ret
        # 0-1 bounds for each weight
        #self.bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
        self.bounds = ((0, 1),) * log_ret.shape[1]
        # Initial Guess (equal distribution)

        self.init_guess = [1/log_ret.shape[1]] * log_ret.shape[1]



    def get_ret_vol_sr(self, weights):
        """
        Takes in log returns and weights, returns array or return,volatility, sharpe ratio
        """
        weights = np.array(weights)
        ret = np.sum(self.log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.log_ret.cov() * 252, weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])


    def neg_sharpe(self, weights):
        return self.get_ret_vol_sr(weights)[2] * -1


    # Constraints
    def check_sum(self, weights):
        """
        Returns 0 if sum of weights is 1.0
        """
        return np.sum(weights) - 1


    def get_weights(self):
        # Sequential Least Squares Programming (SLSQP).
        opt_results = minimize(self.neg_sharpe, self.init_guess, method='SLSQP', bounds=self.bounds, constraints=self.cons)
        return np.array(opt_results.x)


if __name__ == "__main__":
    dataset = MyDataset()
    config = EnvConfig()
    train, _ = dataset.get_train_test()
    log_returns = train[config.stocks]
    sharpe = Sharpe(log_returns)
    w = sharpe.get_weights()
    print(w)

