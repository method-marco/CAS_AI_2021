import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


class SarsaMaxDiscrete:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, bins=(10, 10), alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.state_grid = SarsaMaxDiscrete.create_uniform_grid(env.observation_space.low, env.observation_space.high,
                                                               bins)
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)

        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate  # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon

        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        return tuple(SarsaMaxDiscrete.discretize(state, self.state_grid))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action

    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        state = self.preprocess_state(state)
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                                                                   (reward + self.gamma * max(self.q_table[state]) -
                                                                    self.q_table[self.last_state + (self.last_action,)])

            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action

    @staticmethod
    def discretize(sample, grid):
        """Discretize a sample as per given grid.

        Parameters
        ----------
        sample : array_like
            A single sample from the (original) continuous space.
        grid : list of array_like
            A list of arrays containing split points for each dimension.

        Returns
        -------
        discretized_sample : array_like
            A sequence of integers with the same number of dimensions as sample.
        """

        discretized_samples = []
        for s, b in zip(sample, grid):
            discretized_samples.append(int(np.digitize(s, b)))
        return discretized_samples

    @staticmethod
    def create_uniform_grid(low, high, bins=(10, 10)):
        """Define a uniformly-spaced grid that can be used to discretize a space.

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of the continuous space.
        high : array_like
            Upper bounds for each dimension of the continuous space.
        bins : tuple
            Number of bins along each corresponding dimension.

        Returns
        -------
        grid : list of array_like
            A list of arrays containing split points for each dimension.
        """

        split_points = []
        for l, h, n in zip(low, high, bins):
            split_points.append(np.linspace(l, h, n, endpoint=False)[1:])
        return split_points

    @staticmethod
    def run(agent, env, num_episodes=20000, mode='train'):
        """Run agent in given reinforcement learning environment and return scores."""
        scores = []
        max_avg_score = -np.inf
        for i_episode in range(1, num_episodes + 1):
            # Initialize episode
            state = env.reset()
            action = agent.reset_episode(state)
            total_reward = 0
            done = False

            # Roll out steps until done
            while not done:
                state, reward, done, info = env.step(action)
                total_reward += reward
                action = agent.act(state, reward, done, mode)

            # Save final score
            scores.append(total_reward)

            # Print episode stats
            if mode == 'train':
                if len(scores) > 100:
                    avg_score = np.mean(scores[-100:])
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score

                if i_episode % 100 == 0:
                    print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score),
                          end="")
                    sys.stdout.flush()

        return scores

    @staticmethod
    def plot_scores(scores, rolling_window=100):
        """Plot scores and optional rolling mean using specified window."""
        plt.plot(scores);
        plt.title("Scores");
        rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
        plt.plot(rolling_mean);
        return rolling_mean

    @staticmethod
    def plot_q_table(q_table):
        """Visualize max Q-value for each state and corresponding action."""
        q_image = np.max(q_table, axis=2)       # max Q-value for each state
        q_actions = np.argmax(q_table, axis=2)  # best action for each state

        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(q_image, cmap='jet');
        cbar = fig.colorbar(cax)
        for x in range(q_image.shape[0]):
            for y in range(q_image.shape[1]):
                ax.text(x, y, q_actions[x, y], color='white',
                        horizontalalignment='center', verticalalignment='center')
        ax.grid(False)
        ax.set_title("Q-table, size: {}".format(q_table.shape))
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')

    @staticmethod
    def test_agent(env, agent, number_of_episodes=1000):
        state = env.reset()
        score = 0
        for t in range(number_of_episodes):
            action = agent.act(state, mode='test')
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                print('Score: ', score)
                print('Number of Actions: ', t)
                break
        env.close()
