import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader


class BinaryClassificationTrainer:
    def __init__(self, model, loss_fn, device='cpu'):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device

    def train_batch(self, train_dataloader, optimizer):
        size = len(train_dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(self.device), y.unsqueeze(1).to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def train(self, features, labels, batch_size=16, learning_rate=0.0001, epochs=10, n_test=1000):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        training_data = data_utils.TensorDataset(torch.tensor(features[:-n_test]).float().to(self.device),
                                                 torch.tensor(labels[:-n_test]).float().to(self.device))
        test_data = data_utils.TensorDataset(torch.tensor(features[n_test:]).float().to(self.device),
                                             torch.tensor(labels[n_test:]).float().to(self.device))
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_batch(train_dataloader, optimizer)
            self.test(test_dataloader)
        print("Done!")

    def test(self, test_dataloader):
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(self.device), y.unsqueeze(1).to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                pred = (pred > 0.8).float()
                correct += (pred == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error:\nTest Accuracy: {(100 * correct):>0.1f}%, Avg test loss: {test_loss:>8f} \n")
