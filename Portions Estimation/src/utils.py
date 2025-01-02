# Utility functions (e.g., for saving/loading models, metrics)
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt

class MealDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dish_id = self.data.iloc[idx, 0]
        rgb_path = self.data.iloc[idx, 1]
        rgbd_path = self.data.iloc[idx, 2]
        glycemic_load = self.data.iloc[idx, 3]

        # Load the RGB and RGBD tensors
        rgb_tensor = torch.load(rgb_path) # Shape: [3, H, W]
        rgbd_tensor = torch.load(rgbd_path) # Shape: [4, H, W]

        # Concatenate RGB and RGBD tensors along the channel dimension
        combined_tensor = torch.cat((rgb_tensor, rgbd_tensor), dim=0)  # Shape: [7, H, W]
        return combined_tensor, glycemic_load


def get_data_loaders(csv_file, batch_size=32, transform=None, val_size=0.2, test_size=0.2):
    dataset = MealDataset(csv_file=csv_file, transform=transform)

    # Split proportions
    total_size = len(dataset)
    train_size = int((1 - val_size - test_size) * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size  # The remaining data for testing

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def plot_loss_curve(train_loss, val_loss, num_epochs, save_path=None):
    """Plot training and validation loss curves."""
    x = np.arange(num_epochs)
    plt.plot(x,train_loss, label="Train Loss")
    plt.plot(x, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.xticks(x)
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    if save_path:
        plt.savefig(save_path)
    plt.show()