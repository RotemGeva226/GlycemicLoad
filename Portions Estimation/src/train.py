# Script to train the models
import torch
import torch.optim as optim
import torch.nn as nn
from model import ResNet101WithRGBandRGBD
from utils import get_data_loaders, plot_loss_curve
from tqdm import tqdm

def train():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    csv_file = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\processed\processed_annotations.csv"

    # Prepare data
    train_loader, val_loader, test_loader = get_data_loaders(csv_file, batch_size)

    # Initialize model
    model = ResNet101WithRGBandRGBD()

    # Loss and optimizer
    criterion = nn.MSELoss()  # use MSELoss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()  # Set the model to training mode
        for combined_tensor, glycemic_load in train_loader:
            combined_tensor = combined_tensor.float()
            glycemic_load = glycemic_load.float()

            # Forward pass
            outputs = model(combined_tensor)  # Both inputs are the same tensor

            # Compute loss (predictions, target) # size (batch_size, 1)
            loss = criterion(outputs, glycemic_load)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop - evaluate every few epochs
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for combined_tensor, glycemic_load in val_loader:
                combined_tensor = combined_tensor.float()
                glycemic_load = glycemic_load.float()

                outputs = model(combined_tensor)
                val_loss += criterion(outputs, glycemic_load).item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        losses.append(loss.item())
        val_losses.append(val_loss)

    # Plot loss
    plot_loss_curve(train_loss=losses, val_loss=val_losses, num_epochs=num_epochs)

    # Optionally save the trained model
    saving_option = input('Would you like to save the model?')
    if saving_option.lower() == 'y':
        torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()