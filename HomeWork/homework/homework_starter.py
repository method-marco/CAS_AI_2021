import homework.gradient_descent as gradient_descent
from homework.dataset.SP500ReturnsDataSet import SP500ReturnsDataSet
from homework.dataset.SP500DataSet import SP500DataSet
from homework.binary_classification.BinaryClassificationTrainer import BinaryClassificationTrainer
from homework.binary_classification.NeuronalNetwork import NeuralNetwork
from homework.variational_atuoencoder.VAETrainer import VAETrainer
from homework.variational_atuoencoder.DataAugmenter import DataAugmenter
from homework.forecasting.TFTSP500 import TFTSP500
from homework.reinforcement_learning.monte_carlo.FrozenLakeMC import FrozenLakeMC
from homework.reinforcement_learning.enivronment.TradingEnv import TradingEnv, TradingActions
from homework.reinforcement_learning.sarsa.SarsaMaxDiscrete import SarsaMaxDiscrete
from torch import nn
import gym
import matplotlib.pyplot as plt


def start_homework_gradient_descent():
    plt.figure(figsize=(12, 8))
    gradient_descent.start()
    plt.show()


def start_sp500_binary_classification():
    dataset = SP500ReturnsDataSet()
    data = dataset.load()
    features = data.drop('SPY', axis=1).values
    labels = data.SPY

    model = NeuralNetwork()
    loss_fn = nn.BCEWithLogitsLoss()
    trainer = BinaryClassificationTrainer(model, loss_fn)

    return trainer.train(features, labels)


def start_data_generator_with_vae():
    dataset = SP500ReturnsDataSet()
    data = dataset.load()
    features = data.drop('SPY', axis=1).values
    labels = data.SPY

    trainer_generator = VAETrainer()
    trainer_generator.train(features, labels, epochs=10)

    train_loader, test_loader = dataset.get_loaders()
    generated_data = trainer_generator.generate(train_loader, 100)
    print(generated_data.shape)

    plt.figure(figsize=(12, 8))
    original_first_row = train_loader.dataset.tensors[0].detach().numpy()[0, :]
    generated_first_row = generated_data.to_numpy()[0, :-1]
    plt.plot(original_first_row)
    plt.plot(generated_first_row)
    plt.show()


def start_data_augmenter_sp500():
    dataset = SP500ReturnsDataSet()
    data = dataset.load()
    features = data.drop('SPY', axis=1).values
    labels = data.SPY

    augmenter = DataAugmenter()
    new_f, new_l = augmenter.augment_data(features, labels)
    print('Number of original rows {}'.format(features.shape))
    print('Number of new rows {}'.format(new_f.shape))


def start_sp500_binary_classification_with_augmented_data():
    dataset = SP500ReturnsDataSet()
    data = dataset.load()
    features = data.drop('SPY', axis=1).values
    labels = data.SPY

    augmenter = DataAugmenter(augmented_batch_size=4, augmented_after_batch=16)
    new_features, new_labels = augmenter.augment_data(features, labels, augmentation_strength=1.5)
    # new_features, new_labels = augmenter.append_augment_data(features, labels, augmentation_strength=1)

    model = NeuralNetwork()
    loss_fn = nn.BCEWithLogitsLoss()
    trainer = BinaryClassificationTrainer(model, loss_fn)

    return trainer.train(new_features, new_labels)


def start_sp500_tft():
    tft = TFTSP500()
    tft.load_data()
    tft.create_tft_model()
    tft.train()
    tft.evaluate(number_of_examples=1)
    plt.show()


def start_frozen_lake_mc():
    flmc = FrozenLakeMC()
    flmc.print_environment_info()
    flmc.train(num_episodes=100000)
    mean_score = flmc.evaluate(num_episodes=100)
    print('Mean Score: {}'.format(mean_score))


def start_rl_trading_env():
    stocks = ['AAPL', 'MSFT', 'NFLX', 'AMZN']

    dataset = SP500DataSet(stocks)
    df_data = dataset.load()

    env = TradingEnv(stocks, df_data, None)
    print(env.states)

    n_episodes = 1
    for i in range(n_episodes):
        state = env.reset()
        print(state)
        done = False
        next_action = TradingActions.Hold
        while not done:
            next_state, reward, done = env.step(next_action)
            print('Action: {}, Reward: {}'.format(next_action, reward))
            if reward < 0:
                next_action=TradingActions.Buy
            elif reward < 0:
                next_action=TradingActions.Hold
            else:
                next_action=TradingActions.Sell


def start_rl_mountain_car():
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
    env = gym.make('MountainCar-v0')
    env.seed(505)
    agent = SarsaMaxDiscrete(env, bins=(20, 10))
    scores = SarsaMaxDiscrete.run(agent, env, num_episodes=100000)
    SarsaMaxDiscrete.plot_scores(scores)
    SarsaMaxDiscrete.plot_q_table(agent.q_table)
    SarsaMaxDiscrete.test_agent(env, agent)

def start_rl_cart_pole():
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    env = gym.make('CartPole-v0')
    env.seed(505)
    agent = SarsaMaxDiscrete(env, bins=(20, 10))
    scores = SarsaMaxDiscrete.run(agent, env, num_episodes=100000)
    SarsaMaxDiscrete.plot_scores(scores)
    SarsaMaxDiscrete.plot_q_table(agent.q_table)
    SarsaMaxDiscrete.test_agent(env, agent)
