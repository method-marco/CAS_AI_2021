import homework.gradient_descent as gradient_descent
from homework.dataset.SP500DataSet import SP500DataSet
from homework.binary_classification.BinaryClassificationTrainer import BinaryClassificationTrainer
from homework.binary_classification.NeuronalNetwork import NeuralNetwork
from homework.variational_atuoencoder.VAETrainer import VAETrainer
from homework.variational_atuoencoder.DataAugmenter import DataAugmenter
from homework.forecasting.TFTSP500 import TFTSP500
from homework.reinforcement_learning.monte_carlo.FrozenLakeMC import FrozenLakeMC
from homework.reinforcement_learning.enivronment.TradingEnv import TradingEnv
from torch import nn
import matplotlib.pyplot as plt


def start_homework_gradient_descent():
    plt.figure(figsize=(12, 8))
    gradient_descent.start()
    plt.show()


def start_sp500_binary_classification():
    dataset = SP500DataSet()
    data = dataset.load()
    features = data.drop('SPY', axis=1).values
    labels = data.SPY

    model = NeuralNetwork()
    loss_fn = nn.BCEWithLogitsLoss()
    trainer = BinaryClassificationTrainer(model, loss_fn)

    return trainer.train(features, labels)


def start_data_generator_with_vae():
    dataset = SP500DataSet()
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
    dataset = SP500DataSet()
    data = dataset.load()
    features = data.drop('SPY', axis=1).values
    labels = data.SPY

    augmenter = DataAugmenter()
    new_f, new_l = augmenter.augment_data(features, labels)
    print('Number of original rows {}'.format(features.shape))
    print('Number of new rows {}'.format(new_f.shape))


def start_sp500_binary_classification_with_augmented_data():
    dataset = SP500DataSet()
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
    dataset = SP500DataSet()
    df_data = dataset.load()
    env = TradingEnv(df_data, None)
    state = env.reset()
    print(state)
