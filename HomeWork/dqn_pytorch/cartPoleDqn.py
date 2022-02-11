import numpy as np
import torch
from torch import nn
from torch.optim import Adam, Optimizer


class DQN(nn.Module):
    def __init__(self, state_shape: int, num_actions: int, learning_rate: float):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.internal_model = self.build_model()
        self.optimizer = self.create_optimizer()

    def build_model(self) -> nn.Sequential:
        net = nn.Sequential(
            nn.Linear(int(self.state_shape[0]), 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.num_actions)
        )
        return net

    def create_optimizer(self) -> Optimizer:
        return Adam(self.internal_model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.internal_model(x)

    def fit(self, q_values, expected_q_values):
        # Compute MSE loss
        criterion = nn.MSELoss()
        loss = criterion(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.internal_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_model(self, other_model: nn.Module):
        self.load_state_dict(other_model.state_dict())

    def load_model(self, path: str):
        self.internal_model.load_state_dict(torch.load(path))

    def save_model(self, path: str):
        torch.save(self.internal_model.state_dict(), path)


if __name__ == "__main__":
    dqn = DQN(
        state_shape=4,
        num_actions=2,
        learning_rate=0.001
    )
    dqn.internal_model.summary()
