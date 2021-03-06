import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, n_obs, reparameterization_strength = 1):
        super(VAE, self).__init__()
        self.reparameterization_strength = reparameterization_strength
        self.fc1 = nn.Linear(n_obs, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, n_obs)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = self.reparameterization_strength * torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)  # torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
