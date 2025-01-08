# Script to train the models
import torch
import torch.nn as nn
from model import ResNet34WithRGBandRGBD, ResNet18WithRGBandRGBD
from utils import get_data_loaders, save_model, load_model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



def train(model, experiment_name, batch_size, num_epochs, learning_rate, weight_decay):
    # Initialize general
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs-updated/{experiment_name}")
    model.to(device)
    # Hyperparameters
    # min_delta = 0.001
    # patience = 20
    csv_file = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\processed\processed_annotations.csv"

    # Prepare data
    train_loader, val_loader, test_loader = get_data_loaders(csv_file, batch_size)

    # Loss and optimizer
    criterion = nn.MSELoss()  # use MSELoss for regression
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # # Early stopping mechanism
    # best_val_loss = float('inf')
    # patience_counter = 0

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

        # # Early stopping logic
        # if curr_val_loss < best_val_loss - min_delta:
        #     best_val_loss = curr_val_loss
        #     patience_counter = 0  # Reset patience counter
        # else:
        #     patience_counter += 1
        #     print(f"No improvement in validation loss for {patience_counter} epochs.")
        #     if patience_counter >= patience:
        #         print("Early stopping triggered. Stopping training.")
        #         break

    # Optionally save the trained model
    saving_option = input('Would you like to save the model?')
    if saving_option.lower() == 'y':
        save_model(model, f"trained_models/{experiment_name}.pth", num_epochs, optimizer, loss)
    writer.close()

if __name__ == "__main__":
    model = ResNet18WithRGBandRGBD(is_pretrained=False)
    experiment_name = "untrained_resnet18_batch32_0.00001_lr"
    batch_size = 32
    num_epochs = 800
    learning_rate = 0.00001
    weight_decay = 0.0
    train(model, experiment_name, batch_size, num_epochs, learning_rate, weight_decay)