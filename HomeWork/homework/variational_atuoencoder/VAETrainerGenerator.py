import numpy as np
import pandas as pd
import torch
import os
from homework.variational_atuoencoder.VariationalAutoEncoder import VAE
from homework.dataset.SP500DataSet import SP500DataSet
from torch.nn import functional as F


class VAETrainerGenerator:

    def __init__(self):
        super(VAETrainerGenerator, self).__init__()
        self.model = None

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        torch.manual_seed(0)

        dataset = SP500DataSet()
        train_loader, test_loader = dataset.get_loaders(device=self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.size = train_loader.dataset.tensors[0].data.shape[1]  # + 1

        self.model = VAE(self.size).to(self.device)  # TO DO
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data[0].to(self.device)
            # torch.cat([data[0], ata[1]], dim=1)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def train(self, epochs=10):
        for epoch in range(1, epochs+1):
            self.train_epoch(epoch)
            self.test(epoch)
            # with torch.no_grad():
            #    sample = torch.randn(64, 20).to(trainer.device)
            #    sample = trainer.model.decode(sample).cpu()

        self.save()

        self.model = VAE(self.size)
        self.model.load_state_dict(torch.load('./runs/vae/vae_state_dict'))

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
                data = data[0].to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        # Reconstruction + KL divergence losses summed over all elements and batch

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x.view(-1, self.size), reduction='mean')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

    def generate(self, number_of_datasets=10):
        samples = []
        for i in range(number_of_datasets):
            with torch.no_grad():
                sample_feat, _, _ = self.model(self.train_loader.dataset.tensors[0])
                sample_labels = torch.unsqueeze(self.train_loader.dataset.tensors[1], dim=1)
                new_data_sample = torch.cat([sample_feat, sample_labels], dim=1)
                samples.append(new_data_sample.numpy())
        return pd.DataFrame(np.vstack(samples))