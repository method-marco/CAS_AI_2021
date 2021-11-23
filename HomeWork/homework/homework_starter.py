import homework.gradient_descent as gradient_descent
from homework.dataset.SP500DataSet import SP500DataSet
from homework.binary_classification.Trainer import Trainer
from homework.binary_classification.NeuronalNetwork import NeuralNetwork
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
    trainer = Trainer(model, loss_fn)

    trainer.train(features, labels)
