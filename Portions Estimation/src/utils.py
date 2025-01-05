# Utility functions (e.g., for saving/loading models, metrics)
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torchvision.transforms as T


class MealDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dish_id = self.data.iloc[idx, 0]
        combined_path = self.data.iloc[idx, 1]
        glycemic_load = self.data.iloc[idx, 2]

        # Load the RGB and RGBD tensors
        combined_tensor = torch.load(combined_path) # Shape: [4, H, W]

        # Separate RGB and depth channels
        rgb_tensor = combined_tensor[:3]  # Shape: [3, H, W]
        depth_tensor = combined_tensor[3:]  # Shape: [1, H, W]

        # Apply augmentations to the RGB tensor
        if self.transform:
            augmented_rgb = self.transform(rgb_tensor)

            # Ensure depth tensor matches RGB transformations
            depth_tensor = self.transform(depth_tensor)

        # Combine RGB and depth tensors back
        combined_tensor = torch.cat((augmented_rgb, depth_tensor), dim=0)

        return combined_tensor, glycemic_load


def get_data_loaders(csv_file, batch_size=32, val_size=0.2, test_size=0.2):
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Expects [1 or 3, H , W]
    ])

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
    print(f'The train loss array is: {train_loss}')
    print(f'The validation loss array is: {val_loss}')
    x = np.arange(num_epochs)
    plt.plot(x,train_loss, label="Train Loss")
    plt.plot(x, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.xticks(np.arange(0, num_epochs, 5), fontsize=10)
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    if save_path:
        plt.savefig(save_path)
    plt.show()