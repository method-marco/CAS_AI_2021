from __future__ import print_function
import argparse

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from dataset import MyDataset
from utils import VAEConfig, EnvConfig
import os


class Trainer():

    def __init__ (self):
        super(Trainer, self).__init__()
        self.vae_config = VAEConfig()
        self.env_config = EnvConfig()

        cuda = self.vae_config.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        torch.manual_seed(self.vae_config.seed)

        dataset = MyDataset()
        train_ds, test_ds = dataset.get_train_test()

        train_ds = train_ds.loc[:, ~train_ds.columns.isin(self.env_config.stocks_adj_close_names)].to_numpy()
        train_ds = torch.tensor(train_ds).float()

        test_ds = test_ds.loc[:, ~test_ds.columns.isin(self.env_config.stocks_adj_close_names)].to_numpy()
        test_ds = torch.tensor(test_ds).float()

        self.n_obs = test_ds.shape[1]

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.vae_config.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=self.vae_config.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.vae_config.batch_size, shuffle=True, **kwargs)

        self.model = VAE(self.n_obs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.vae_config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def save(self):
        path = './runs/{}/'.format('vae')
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.model.state_dict(), os.path.join(path, 'vae_state_dict'))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        # Reconstruction + KL divergence losses summed over all elements and batch

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x.view(-1, self.n_obs), reduction='mean')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD


class VAE(nn.Module):
    def __init__(self, n_obs):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n_obs, 400)
        self.fc21 = nn.Linear(400, 20) # mu
        self.fc22 = nn.Linear(400, 20) # log varianz
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, n_obs)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3) # torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar





if __name__ == "__main__":
    trainer = Trainer()

    for epoch in range(1, trainer.vae_config.epochs + 1):
        trainer.train(epoch)
        trainer.test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(trainer.device)
            sample = trainer.model.decode(sample).cpu()

        trainer.save()

    model = VAE(trainer.n_obs)
    model.load_state_dict(torch.load('./runs/vae/vae_state_dict'))

