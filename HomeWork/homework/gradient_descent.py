import matplotlib.pyplot as plt
import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.history_m = []
        self.history_b = []

    def update_weights(self, m, b, x, y, learning_rate):
        m_deriv = 0
        b_deriv = 0
        N = len(x)
        # Ableitung MSE
        for i in range(N):  # pytorch - backward # gradient tape
            # Calculate partial derivatives
            # -x (innere Ableitung) * 2(y - (mx + b)) (äussere Ableitung)
            m_deriv += -2 * x[i] * (y[i] - (m * x[i] + b))

            # -2(y - (mx + b))
            b_deriv += -2 * (y[i] - (m * x[i] + b))

        # We subtract because the derivatives point in direction of steepest ascent
        m -= (m_deriv / float(N)) * learning_rate  # später - in pytorch step
        b -= (b_deriv / float(N)) * learning_rate
        self.history_m.append(m)
        self.history_b.append(b)

        return m, b


class GradientDescent:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def train(self, epochs, x, y):
        m = 0
        b = 0
        learning_rate = 0.01
        for i in range(epochs):
            # abbruchskriterium - "early stopping"
            m, b = self.loss_function.update_weights(m, b, x, y, learning_rate)
        return m, b


def train_and_display(x, y, a, b, loss_function=MeanSquaredError(), epochs=1000):
    gradient_descent = GradientDescent(loss_function=loss_function)
    gradient_descent.train(epochs, x, y)

    plt.plot(gradient_descent.loss_function.history_m[0:epochs])
    plt.plot(gradient_descent.loss_function.history_b[0:epochs])

    plt.axhline(y=a, xmin=0, xmax=epochs, c='r', linewidth=4, linestyle='--')
    plt.axhline(y=b, xmin=0, xmax=epochs, c='b', linewidth=4, linestyle=':')
    plt.ylabel('m', fontsize=18)
    plt.xlabel('b', fontsize=18)
    plt.legend(['m', 'b'], loc='upper right', fontsize=18)


def start():
    for i in range(3):
        X = np.random.rand(1000)
        a = 4
        b = 2
        Y = [a * x + b for x in X]  # ein Axon, immer nur eins, Function -

        # ab wann feuert er (hier sofort), wie werden die Inputs moduliert (hier mit 2*x + 6)
        # aktivierungs-funktion(Y), Schalter ein/aus
        train_and_display(X, Y, a, b, loss_function=MeanSquaredError(), epochs=2000)
