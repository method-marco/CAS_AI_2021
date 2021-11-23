import homework.gradient_descent as gradient_descent
from homework.dataset.SP500DataSet import SP500DataSet
from homework.binary_classification.BinaryClassificationTrainer import BinaryClassificationTrainer
from homework.binary_classification.NeuronalNetwork import NeuralNetwork
from homework.variational_atuoencoder.VAETrainer import VAETrainer
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

    trainer.train(features, labels)


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
