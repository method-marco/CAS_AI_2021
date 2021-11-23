import numpy as np
import pandas as pd
import torch
import os
from homework.variational_atuoencoder.VariationalAutoEncoder import VAE
from torch.nn import functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader


class VAETrainer:

    def __init__(self, reparameterization_strength = 1):
        super(VAETrainer, self).__init__()
        torch.manual_seed(0)

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.reparameterization_strength = reparameterization_strength
        self.model = None

    def train_epoch(self, train_dataloader, optimizer, epoch):
        size = train_dataloader.dataset.tensors[0].data.shape[1]  # + 1
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_dataloader):
            data = data[0].to(self.device)
            # torch.cat([data[0], ata[1]], dim=1)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar, size)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                           100. * batch_idx / len(train_dataloader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset)))

    def train(self, features, labels, batch_size=16, learning_rate=1e-3, epochs=10, n_test=1000, shuffle=False):
        training_data = data_utils.TensorDataset(torch.tensor(features[:-n_test]).float().to(self.device),
                                                 torch.tensor(labels[:-n_test]).float().to(self.device))
        test_data = data_utils.TensorDataset(torch.tensor(features[n_test:]).float().to(self.device),
                                             torch.tensor(labels[n_test:]).float().to(self.device))
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        size = train_dataloader.dataset.tensors[0].data.shape[1]  # + 1
        self.model = VAE(size, self.reparameterization_strength).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            self.train_epoch(train_dataloader, optimizer, epoch)
            self.test(test_dataloader, epoch)
            # with torch.no_grad():
            #    sample = torch.randn(64, 20).to(trainer.device)
            #    sample = trainer.model.decode(sample).cpu()

        self.save()

        self.model = VAE(size).to(self.device)
        self.model.load_state_dict(torch.load('./runs/vae/vae_state_dict'))

    def save(self):
        path = './runs/{}/'.format('vae')
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.model.state_dict(), os.path.join(path, 'vae_state_dict'))

    def test(self, test_dataloader, epoch):
        size = test_dataloader.dataset.tensors[0].data.shape[1]  # + 1
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                data = data[0].to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar, size).item()

        test_loss /= len(test_dataloader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        # Reconstruction + KL divergence losses summed over all elements and batch

    def loss_function(self, recon_x, x, mu, logvar, size):
        MSE = F.mse_loss(recon_x, x.view(-1, size), reduction='mean')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

    def generate(self, train_dataloader, number_of_datasets=10):
        samples = []
        for i in range(number_of_datasets):
            with torch.no_grad():
                sample_feat, _, _ = self.model(train_dataloader.dataset.tensors[0])
                sample_labels = torch.unsqueeze(train_dataloader.dataset.tensors[1], dim=1)
                new_data_sample = torch.cat([sample_feat, sample_labels], dim=1)
                samples.append(new_data_sample.numpy())
        return pd.DataFrame(np.vstack(samples))
