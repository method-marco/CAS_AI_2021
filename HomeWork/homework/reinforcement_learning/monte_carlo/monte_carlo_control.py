import sys
import gym
import numpy as np
from collections import defaultdict


def generate_episode_from_Q(env, Q, epsilon, nA):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    while True:
        if state in Q:
            probs = get_probs(Q[state], epsilon, nA)
            # print('Probs: {} for State: {}'.format(probs, state))
            action = np.random.choice(np.arange(nA), p=probs)
        else:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def get_probs(Q_s, epsilon, nA):  # epsilon greedy
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    # Q_s: Returns for state
    # epsilon: will be reduced over time --> the smaller it gets, the higher the probability of best_a according to Q_s
    # for epsilon = 1 all action have same probability
    # nA: Number of actions
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s


def update_Q(env, episode, Q, alpha, gamma):  # policy improvement oder control
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
    for i, state in enumerate(states):
        n_steps_after_state = len(rewards[i:])
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha * (sum(rewards[i:] * discounts[:n_steps_after_state]) - old_Q)
        # print('state:', state)
        # print('action:', actions[i])
        # print('q-value:', Q[state][actions[i]])
    return Q


def mc_prediction_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}. Epsilon: {}".format(i_episode, num_episodes, epsilon), end="")
            sys.stdout.flush()
        # set the value of epsilon
        # Epsilon wird minimiert, dadurch wird immer mehr die gelernte policy befolgt
        epsilon = max(epsilon * eps_decay, eps_min)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha,
                     gamma)  # prediction und eigentllich control, weil Q table die Vorlage für die nächste Action Selection ist
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((state, np.argmax(actions)) for state, actions in Q.items())  # control - Q table wird gelesen
    print()
    return policy, Q


def run_episode(env, policy, gamma=1.0, render=False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        if obs in policy:
            action = int(policy[obs])
        else:
            action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)
