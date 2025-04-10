# Utility functions (e.g., for saving/loading models, metrics)
import os.path

import torch
from torch.utils.data import DataLoader

from data.MealDataset import MealDataset


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

def load_model(model_path, model_class, model_args: dict=None):
    try:
        model = model_class(**model_args)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    except Exception as e:
        print('An error occurred while loading the model:', e)

def load_test_loader(batch_size, tl_name):
    tl_folderpath = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\PortionsEstimation\src\models/saved/tl-single_ingr_portions_regression"
    csv_file = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\PortionsEstimation\src\data\processed\processed_single_ingr_portions_regression\processed_single_ingr_portions_regression.csv"
    dataset = MealDataset(csv_file)
    train_loader, val_loader, test_loader = dataset.get_regression_loader(batch_size=batch_size)
    torch.save(test_loader.dataset,os.path.join(tl_folderpath, tl_name + ".pt"))
    test_dataset = torch.load(os.path.join(tl_folderpath, tl_name + ".pt"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

if __name__ == "__main__":
    load_model(
        model_path=r"/Portions Estimation/src/trained_models_regression\Test1.pth",
        model_class=ResNet34WithRGBandRGBD,
        model_args={'is_pretrained': False,}, optimizer_class=torch.optim.AdamW, optimizer_args={'lr': 1e-4})