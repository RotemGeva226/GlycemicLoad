# Script to train the models
import torch
import torch.nn as nn
from model import ResNet34WithRGBandRGBD, ResNet50WithRGBandRGBD
from utils import get_data_loaders, save_model, load_model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



def train(model, experiment_name, batch_size, num_epochs, learning_rate, weight_decay):
    """
    This function trains the model.
    :param model: the model to be trained.
    :param experiment_name: the name of the experiment.
    :param batch_size: the batch size.
    :param num_epochs: the number of epochs.
    :param learning_rate: the learning rate.
    :param weight_decay: the weight decay.
    """
    # Initialize general
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs-updated/{experiment_name}")
    model.to(device)
    csv_file = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\processed\processed_annotations.csv"

    # Prepare data
    train_loader, val_loader, test_loader = get_data_loaders(csv_file, batch_size)
    torch.save(test_loader, f"trained_models/tl-{experiment_name}.pt")

    # Loss and optimizer
    criterion = nn.MSELoss()  # use MSELoss for regression
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()  # Set the model to training mode
        for combined_tensor, glycemic_load in train_loader:
            combined_tensor = combined_tensor.float().to(device)
            glycemic_load = glycemic_load.float().to(device)

            # Forward pass
            outputs = model(combined_tensor)  # Both inputs are the same tensor

            # Compute loss (predictions, target) # size (batch_size, 1)
            loss = criterion(outputs, glycemic_load)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for combined_tensor, glycemic_load in val_loader:
                combined_tensor = combined_tensor.float().to(device)
                glycemic_load = glycemic_load.float().to(device)

                outputs = model(combined_tensor)
                val_loss += criterion(outputs, glycemic_load).item()

        curr_loss = loss.item()
        curr_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {curr_loss:.4f}, Val Loss: {curr_val_loss:.4f}")   # Average validation loss per batch

        # Write to Tensorboard
        writer.add_scalar('Loss/train', curr_loss, epoch)
        writer.add_scalar('Loss/validation', curr_val_loss, epoch)

    # Optionally save the trained model
    saving_option = input('Would you like to save the model?')
    if saving_option.lower() == 'y':
        save_model(model, f"trained_models/{experiment_name}.pth", num_epochs, optimizer, loss)
    writer.close()

def continue_train(model, experiment_name, batch_size, num_epochs, epochs, loss, optimizer):
    """
    This function trains the model from a certain epoch.
    :param model: the model to be trained.
    :param experiment_name: the name of the experiment.
    :param batch_size: the batch size.
    :param num_epochs: the number of epochs.
    :param epochs: the number of epochs that have already been trained.
    :param loss: the loss.
    :param optimizer: the optimizer.
    """
    # Initialize general
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs-updated/{experiment_name}")
    model.to(device)
    csv_file = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\processed\processed_annotations.csv"

    # Prepare data
    train_loader, val_loader, test_loader = get_data_loaders(csv_file, batch_size)

    # Loss and optimizer
    criterion = nn.MSELoss()  # use MSELoss for regression

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()  # Set the model to training mode
        for combined_tensor, glycemic_load in train_loader:
            combined_tensor = combined_tensor.float().to(device)
            glycemic_load = glycemic_load.float().to(device)

            # Forward pass
            outputs = model(combined_tensor)

            # Compute loss (predictions, target) # size (batch_size, 1)
            loss = criterion(outputs, glycemic_load)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for combined_tensor, glycemic_load in val_loader:
                combined_tensor = combined_tensor.float().to(device)
                glycemic_load = glycemic_load.float().to(device)

                outputs = model(combined_tensor)
                val_loss += criterion(outputs, glycemic_load).item()

        curr_loss = loss.item()
        curr_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epochs+epoch+1}/{num_epochs+epochs}], Loss: {curr_loss:.4f}, Val Loss: {curr_val_loss:.4f}")   # Average validation loss per batch

        # Write to Tensorboard
        writer.add_scalar('Loss/train', curr_loss, epochs+epoch+1)
        writer.add_scalar('Loss/validation', curr_val_loss, epochs+epoch+1)

    # Optionally save the trained model
    saving_option = input('Would you like to save the model?')
    if saving_option.lower() == 'y':
        save_model(model, f"trained_models/{experiment_name}.pth", num_epochs, optimizer, loss)
    writer.close()


if __name__ == "__main__":
    train(ResNet18WithRGBandRGBD(is_pretrained=False), 'untrained_resnet18_batch32_0.00001lr-new2',batch_size=32, num_epochs=1700, learning_rate=0.00001, weight_decay=0.0)