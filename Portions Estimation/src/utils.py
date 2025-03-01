# Utility functions (e.g., for saving/loading models, metrics)
from collections import Counter

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torchvision.transforms as T
from model_regression import ResNet34WithRGBandRGBD
from torch.utils.data.sampler import WeightedRandomSampler



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
        combined_tensor = torch.load(combined_path, weights_only=False) # Shape: [4, H, W]

        # Separate RGB and depth channels
        rgb_tensor = combined_tensor[:3]  # Shape: [3, H, W]
        depth_tensor = combined_tensor[3:]  # Shape: [1, H, W]

        # Apply augmentations to the RGB tensor
        if self.transform:
            augmented_rgb = self.transform(rgb_tensor)

            depth_tensor = self.transform(depth_tensor)

        # Combine RGB and depth tensors back
        combined_tensor = torch.cat((augmented_rgb, depth_tensor), dim=0)

        return combined_tensor, glycemic_load

class MealDatasetClassification(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dish_id = self.data.iloc[idx, 0]
        input_path = self.data.iloc[idx, 1]
        classification = self.data.iloc[idx, 2]

        # Load the RGB tensor
        rgb_tensor = torch.load(input_path, weights_only=False)

        return rgb_tensor, classification

def get_data_loaders(dataset_class, dataset_args=None, batch_size=32, val_size=0.2, test_size=0.2):
    dataset = dataset_class(**dataset_args)

    # Split into different sets
    total_size = len(dataset)
    train_size = int((1 - val_size - test_size) * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size  # The remaining data for testing

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Calculate targets for the training set only
    train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_counts = list(Counter(train_targets).values())  # Counts for each class - what is the freq of each class
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float) # less frequent classes get higher weights
    train_sample_weights = [class_weights[t] for t in train_targets]
    # Replacement = True, allows samples to be picked more than once in an epoch

    # Calculating tergets for validation set
    val_targets = [val_dataset[i][1] for i in range(len(val_dataset))]
    val_sample_weights = [class_weights[t] for t in val_targets]

    # The sampler will oversample minority classes during training
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )
    val_sampler = WeightedRandomSampler(
        weights=val_sample_weights,
        num_samples=len(val_sample_weights),
        replacement=True
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                             num_workers=4, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, class_weights

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

def save_model(model, model_path, epoch, optimizer, loss):
    try:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, model_path)
        print('Model saved successfully.')
    except Exception as e:
        print(f'An error occurred while saving the model: {e}')

def load_model(model_path, model_class, optimizer_class=None, model_args: dict=None, optimizer_args: dict=None, mode='continue_training'):
    try:
        model = model_class(**model_args)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if mode == 'continue_training':
            optimizer = optimizer_class(model.parameters(), **optimizer_args)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print('Model loaded successfully.')
            return model, optimizer, epoch, loss
        else:
            return model
    except Exception as e:
        print('An error occurred while loading the model:', e)

if __name__ == "__main__":
    load_model(
        model_path=r"/Portions Estimation/src/trained_models_regression\Test1.pth",
        model_class=ResNet34WithRGBandRGBD,
        model_args={'is_pretrained': False,}, optimizer_class=torch.optim.AdamW, optimizer_args={'lr': 1e-4})