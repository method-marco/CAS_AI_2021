import gym
from homework.reinforcement_learning.monte_carlo.monte_carlo_control import mc_prediction_control, evaluate_policy


class FrozenLakeMC:

    def __init__(self):
        # https://reinforcement-learning4.fun/2019/06/09/introduction-reinforcement-learning-frozen-lake-example/
        self.env = gym.make('FrozenLake-v1', is_slippery=False)
        self.policy = None

    def print_environment_info(self):
        print('Observations: {}'.format(self.env.observation_space))
        print('Actions: {}'.format(self.env.action_space))

    def train(self, num_episodes=100000, alpha=0.02):
        policy, Q = mc_prediction_control(self.env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05)
        self.policy = policy
        return policy

    def evaluate(self):
        if not self.policy:
            print('Train first!')
            return 'Train first!'
        return evaluate_policy(self.env, self.policy)


