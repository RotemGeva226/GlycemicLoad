# Script to train the models
import torch
import torch.nn as nn
from model_classification import ResNet18
from utils import get_data_loaders, save_model, MealDatasetClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def train_regression(model, experiment_name, batch_size, num_epochs, learning_rate, weight_decay):
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

def train_classification(model, experiment_name, batch_size, num_epochs, learning_rate, weight_decay):
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
    csv_file = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\rgb_processed_imagenet\processed_annotations.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs-classification/{experiment_name}")
    model.to(device)

    # Prepare data
    train_loader, val_loader, test_loader = get_data_loaders(dataset_class=MealDatasetClassification, dataset_args={"csv_file": csv_file})
    torch.save(test_loader, f"trained_models/tl-{experiment_name}.pt")
    class_counts = torch.tensor([250, 250, 173])
    class_weights = 1. / class_counts
    class_weights /= class_weights.sum()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() # use cross entropy for multi class classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()  # Set the model to training mode
        for input_tensor, classification in train_loader:
            input_tensor = input_tensor.float().to(device)
            classification = classification.float().to(device)

            # Forward pass
            outputs = model(input_tensor)  # Both inputs are the same tensor

            # Compute loss (predictions, target) # size (batch_size, 1)
            loss = criterion(outputs, classification)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for input_tensor, classification in val_loader:
                input_tensor = input_tensor.float().to(device)
                classification = classification.long().to(device)

                outputs = model(input_tensor)
                val_loss += criterion(outputs, classification).item()

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(classification.cpu().numpy())

                total += classification.size(0)
                correct += (predicted == classification).sum().item()

        accuracy = 100 * correct / total
        curr_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {curr_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
        disp.plot(cmap="Blues")
        disp.figure_.savefig(f"confusion_matrix_{experiment_name}.png")

        # Write to Tensorboard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Loss/validation', curr_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)

    # Save the trained model
    save_model(model, f"trained_models_classification/{experiment_name}.pth", num_epochs, optimizer, loss)
    writer.close()



if __name__ == "__main__":
    train_classification(ResNet18(num_classes=3), 'resnet18_batch32_0.0001lr',batch_size=32, num_epochs=1000, learning_rate=0.0001, weight_decay=0.0)