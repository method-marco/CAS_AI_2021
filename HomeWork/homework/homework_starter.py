import homework.gradient_descent as gradient_descent
from homework.dataset.SP500DataSet import SP500DataSet
from homework.binary_classification.BinaryClassificationTrainer import BinaryClassificationTrainer
from homework.binary_classification.NeuronalNetwork import NeuralNetwork
from homework.variational_atuoencoder.VAETrainerGenerator import VAETrainerGenerator
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
    trainer_generator = VAETrainerGenerator()
    trainer_generator.train(10)
    generated_data = trainer_generator.generate(10)

    plt.figure(figsize=(12, 8))
    original_np = trainer_generator.train_loader.dataset.tensors[0].detach().numpy()[0, :]
    generated_np = generated_data.to_numpy()[0, :-1]
    plt.plot(original_np[:100])
    plt.plot(generated_np[:100])
    plt.show()
