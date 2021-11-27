from homework.variational_atuoencoder.VAETrainer import VAETrainer
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import torch


class DataAugmenter:
    def __init__(self, augmented_after_batch=5, augmented_batch_size=16, vae_batch_size=16, vae_learning_rate=1e-3,
                 vae_epochs=10):
        self.augmented_after_batch = augmented_after_batch
        self.augmented_batch_size = augmented_batch_size

        self.vae_batch_size = vae_batch_size
        self.vae_learning_rate = vae_learning_rate
        self.vae_epochs = vae_epochs

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

    def augment_data(self, features, labels, augmentation_strength=1, replace=False):
        trainer = VAETrainer(augmentation_strength)
        trainer.train(features, labels, epochs=self.vae_epochs, batch_size=self.vae_batch_size,
                      learning_rate=self.vae_learning_rate)

        data = data_utils.TensorDataset(torch.tensor(features).float().to(self.device),
                                        torch.tensor(labels).float().to(self.device))
        dataloader = DataLoader(data, batch_size=self.augmented_batch_size, shuffle=False)

        new_features = None
        new_labels = None
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                x, y = data
                create_augmented = batch_idx % self.augmented_after_batch == 0
                if create_augmented:
                    x_augmented, _, _ = trainer.model(x)

                if new_features is None:
                    new_features = x
                    new_labels = y
                else:
                    if not replace and create_augmented:
                        new_features = torch.cat([new_features, x_augmented], dim=0)
                        new_labels = torch.cat([new_labels, y], dim=0)
                        new_features = torch.cat([new_features, x], dim=0)
                        new_labels = torch.cat([new_labels, y], dim=0)
                    else:
                        x_to_add = x
                        if create_augmented:
                            x_to_add = x_augmented
                        new_features = torch.cat([new_features, x_to_add], dim=0)
                        new_labels = torch.cat([new_labels, y], dim=0)

            return new_features.numpy(), new_labels.numpy()

    def append_augment_data(self, features, labels, data_multiplication = 2,  augmentation_strength=1):
        trainer = VAETrainer(augmentation_strength)
        trainer.train(features, labels, epochs=self.vae_epochs, batch_size=self.vae_batch_size,
                          learning_rate=self.vae_learning_rate)
        new_features = torch.tensor(features).float().to(self.device)
        new_labels = torch.tensor(labels).float().to(self.device)
        for i in range(data_multiplication):
            with torch.no_grad():
                sample_feat, _, _ = trainer.model(torch.tensor(features).float().to(self.device))
                sample_labels = torch.tensor(labels).float().to(self.device)
                new_features = torch.cat([new_features, sample_feat], dim=0)
                new_labels = torch.cat([new_labels, sample_labels], dim=0)
        return new_features.numpy(), new_labels.numpy()
