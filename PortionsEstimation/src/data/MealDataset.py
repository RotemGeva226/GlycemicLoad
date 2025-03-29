from collections import Counter

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data.sampler import WeightedRandomSampler

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class MealDataset(Dataset):
    def __init__(self, csv_file, train_mode=True):
        self.data = pd.read_csv(csv_file)
        self.train_mode = train_mode

        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15),
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.eval_transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.transform = self.train_transform if train_mode else self.eval_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dish_id = self.data.iloc[idx, 0]
        image_path = self.data.iloc[idx, 1]
        target_value = self.data.iloc[idx, 2]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, target_value


def get_regression_loader(dataset_class, dataset_args=None, batch_size=32, val_size=0.2, test_size=0.2):
    dataset_args['train_mode'] = True
    dataset = dataset_class(**dataset_args)

    # Split into different sets
    total_size = len(dataset)
    train_size = int((1 - val_size - test_size) * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size  # The remaining data for testing

    # Perform the split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                            [train_size, val_size, test_size],
                                                            generator=generator)

    val_dataset.dataset.train_mode = False
    test_dataset.dataset.train_mode = False
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             num_workers=2, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_classification_loader(dataset_class, dataset_args=None, batch_size=32, val_size=0.2, test_size=0.2):
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
                                               num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                             num_workers=2, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, class_weights
