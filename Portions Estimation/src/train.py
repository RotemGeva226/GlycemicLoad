# Script to train the models
import numpy as np
import torch
import torch.nn as nn
from model_classification import ResNet18, ResNet34
from utils import get_data_loaders, save_model, MealDatasetClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score

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
    writer = SummaryWriter(f"runs-regression/{experiment_name}")
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

def train_classification(csv_file, model, experiment_name, batch_size, num_epochs, learning_rate, weight_decay):
    """
    This function trains the model.
    :param csv_file: the file that contains dataset content.
    :param model: the model to be trained.
    :param experiment_name: the name of the experiment.
    :param batch_size: the batch size.
    :param num_epochs: the number of epochs.
    :param learning_rate: the learning rate.
    :param weight_decay: the weight decay.
    """
    # Initialize general
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs-portions-classification/{experiment_name}")
    model.to(device)

    # Prepare data
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(dataset_class=MealDatasetClassification,
                                                                            batch_size=batch_size,
                                                                            dataset_args={"csv_file": csv_file})
    torch.save(test_loader, f"models/trained_models_classification/tl-{experiment_name}.pt")

    # Loss and optimizer
    class_weights = (class_weights / class_weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # use cross entropy for multi class classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()  # Set the model to training mode
        for input_tensor, classification in train_loader:
            input_tensor = input_tensor.float().to(device)
            classification = classification.to(device)

            # Forward pass
            outputs = model(input_tensor)

            # Compute loss (predictions, target)
            loss = criterion(outputs, classification)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []  # For storing probabilities needed for AUC

        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for input_tensor, classification in val_loader:
                input_tensor = input_tensor.float().to(device)
                classification = classification.to(device)

                outputs = model(input_tensor)
                val_loss += criterion(outputs, classification).item()

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(classification.cpu().numpy())

                # Store probabilities for AUC calculation
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

                total += classification.size(0)
                correct += (predicted == classification).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total
        curr_val_loss = val_loss / len(val_loader)

        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Calculate F1 score
        f1 = f1_score(all_targets, all_preds, average='weighted')

        # Calculate AUC (for multi-class, we use one-vs-rest approach)
        # For 3 classes, this will return 3 AUC values
        n_classes = len(np.unique(all_targets))
        auc_scores = []

        # Calculate AUC for each class
        for i in range(n_classes):
            # One-vs-rest approach: current class vs all others
            y_true_binary = (all_targets == i).astype(int)
            y_score = all_probs[:, i]  # Probability for class i
            try:
                auc = roc_auc_score(y_true_binary, y_score)
                auc_scores.append(auc)
            except ValueError:
                # This can happen if a class is not present in the validation set
                auc_scores.append(float('nan'))

        # Average AUC across all classes
        mean_auc = np.nanmean(auc_scores)

        print(f"Val Loss: {curr_val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}, AUC: {mean_auc:.4f}")

        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
        disp.plot(cmap="Blues")
        disp.figure_.savefig(f"confusion_matrix_{experiment_name}.png")

        # Write to Tensorboard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Loss/validation', curr_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)
        writer.add_scalar('F1/validation', f1, epoch)
        writer.add_scalar('AUC/validation', mean_auc, epoch)

    # Save the trained model
    save_model(model, f"models/trained_portions_classification/{experiment_name}.pth", num_epochs, optimizer, loss)
    writer.close()

if __name__ == "__main__":
    csv_filepath = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\processed_portions_classification\processed_portions_classification.csv"
    train_classification(csv_filepath, ResNet34(num_classes=3), 'portions_resnet34_batch64_lr0.0001_wd0.001',
                         batch_size=64, num_epochs=500, learning_rate=0.0001, weight_decay=0.001)