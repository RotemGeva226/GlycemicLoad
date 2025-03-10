import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from utils import save_model, MealDatasetClassification, get_data_loaders
import matplotlib.pyplot as plt
from model_classification import ResNet34, ResNet18, ResNet50


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
    # Initialize W&B
    wandb.init(
        project="meal-portions-classification",
        name=experiment_name,
        config={
            "model_type": model.__class__.__name__,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dataset": csv_file
        }
    )

    # Initialize general
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Log model architecture
    wandb.watch(model, log="all")

    # Prepare data
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(dataset_class=MealDatasetClassification,
                                                                            batch_size=batch_size,
                                                                            dataset_args={"csv_file": csv_file})
    torch.save(test_loader, f"models/portions_classification/tl-{experiment_name}.pt")

    # Loss and optimizer
    class_weights = (class_weights / class_weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # use cross entropy for multi class classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    best_val_loss = float('inf')
    best_accuracy = 0.0

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for input_tensor, classification in train_loader:
            input_tensor = input_tensor.float().to(device)
            classification = classification.to(device)

            # Forward pass
            outputs = model(input_tensor)

            # Compute loss (predictions, target)
            loss = criterion(outputs, classification)
            train_loss += loss.item()
            train_batches += 1

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / train_batches

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

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {curr_val_loss:.4f}, "
              f"Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}, AUC: {mean_auc:.4f}")

        # Create and log confusion matrix as an image
        cm = confusion_matrix(all_targets, all_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
        disp.plot(cmap="Blues", ax=ax)

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": curr_val_loss,
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": mean_auc,
            "confusion_matrix": wandb.Image(fig),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        plt.close(fig)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, f"models/portions_classification/{experiment_name}_best_acc.pth", epoch, optimizer, loss)
            wandb.run.summary["best_accuracy"] = best_accuracy
            wandb.run.summary["best_accuracy_epoch"] = epoch

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            save_model(model, f"models/portions_classification/{experiment_name}_best_loss.pth", epoch, optimizer, loss)
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_val_loss_epoch"] = epoch

    # Save the final trained model
    save_model(model, f"models/portions_classification/{experiment_name}.pth", num_epochs, optimizer, loss)

    # Log model as an artifact
    model_artifact = wandb.Artifact(f"model-{experiment_name}", type="model")
    model_artifact.add_file(f"models/portions_classification/{experiment_name}.pth")
    wandb.log_artifact(model_artifact)

    # Close all
    wandb.finish()


def train_classification_sweep():
    """
    Training function for WandB sweep.
    Gets hyperparameters from wandb.config
    """
    # Initialize wandb
    wandb.init()

    # Get hyperparameters from wandb.config
    config = wandb.config

    # Get model
    model = create_model(
        model_type=config.model_type)

    wandb.run.name = (f"sweep-run-{config.model_type}-{config.batch_size}-"
                       f"{config.learning_rate}--{config.weight_decay}")
    wandb.run.save()

    wandb.summary["model_type"] = config.model_type

    # Fixed parameters
    csv_file = (r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data"
                r"\processed_portions_classification\processed_portions_classification.csv")
    experiment_name = (f"sweep-run-{config.model_type}-{config.batch_size}-"
                       f"{config.learning_rate}--{config.weight_decay}")

    # Initialize general
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Log model architecture
    wandb.watch(model, log="all")

    # Prepare data
    # train_loader, val_loader, test_loader, class_weights = get_data_loaders(
    #     dataset_class=MealDatasetClassification,
    #     batch_size=config.batch_size,
    #     dataset_args={"csv_file": csv_file}
    # )
    # torch.save(test_loader, f"models/portions_classification/tl-{experiment_name}.pt")
    train_loader, val_loader, test_loader, class_weights = prepare_and_log_datasets(csv_file, experiment_name)

    # Loss and optimizer
    class_weights = (class_weights / class_weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Training loop
    best_val_loss = float('inf')
    best_accuracy = 0.0

    for epoch in tqdm(range(config.num_epochs), desc="Training Epochs"):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for input_tensor, classification in train_loader:
            input_tensor = input_tensor.float().to(device)
            classification = classification.to(device)

            # Forward pass
            outputs = model(input_tensor)

            # Compute loss
            loss = criterion(outputs, classification)
            train_loss += loss.item()
            train_batches += 1

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / train_batches

        # Validation loop
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

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

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {curr_val_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}, AUC: {mean_auc:.4f}")

        # Create and log confusion matrix as an image
        cm = confusion_matrix(all_targets, all_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
        disp.plot(cmap="Blues", ax=ax)

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": curr_val_loss,
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": mean_auc,
            "confusion_matrix": wandb.Image(fig),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        plt.close(fig)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, f"models/portions_classification/{experiment_name}_best_acc.pth", epoch, optimizer, loss)
            wandb.run.summary["best_accuracy"] = best_accuracy
            wandb.run.summary["best_accuracy_epoch"] = epoch

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            save_model(model, f"models/portions_classification/{experiment_name}_best_loss.pth", epoch, optimizer, loss)
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_val_loss_epoch"] = epoch

    # Save the final trained model
    save_model(model, f"models/portions_classification/{experiment_name}.pth", config.num_epochs, optimizer, loss)

    # Log model as an artifact
    model_artifact = wandb.Artifact(f"model-{experiment_name}", type="model")
    model_artifact.add_file(f"models/portions_classification/{experiment_name}.pth")
    wandb.log_artifact(model_artifact)


def create_model(model_type):
    if model_type == "ResNet18":
        return ResNet18(num_classes=3)
    elif model_type == "ResNet34":
        return ResNet34(num_classes=3)
    elif model_type == "ResNet50":
        return ResNet50(num_classes=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def prepare_and_log_datasets(csv_file, experiment_name):
    # Create dataset artifact
    dataset_artifact = wandb.Artifact(
        name="meal-classification-dataset",
        type="dataset",
        description="Dataset for meal portion classification",
        metadata={"source": csv_file}
    )

    # Add the raw CSV file
    dataset_artifact.add_file(csv_file)

    # Log the dataset artifact
    wandb.log_artifact(dataset_artifact)

    # Prepare your data loaders
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        dataset_class=MealDatasetClassification,
        batch_size=wandb.config.batch_size,
        dataset_args={"csv_file": csv_file}
    )

    # Save and log the test set separately
    path = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\src\models\portions_classification"
    test_set_path = f"{path}/test_set_{experiment_name}.pt"
    torch.save(test_loader, test_set_path)

    test_artifact = wandb.Artifact(
        name="test-dataset",
        type="dataset",
        description="Test dataset for evaluation",
        metadata={"parent_dataset": csv_file}
    )
    test_artifact.add_file(test_set_path)
    wandb.log_artifact(test_artifact)

    return train_loader, val_loader, test_loader, class_weights

def create_sweep_config():
    sweep_config = {
        'method': 'bayes',  # You can use 'grid', 'random', or 'bayes'
        'metric': {
            'name': 'accuracy',  # The metric to optimize
            'goal': 'maximize'   # We want to maximize accuracy
        },
        'parameters': {
            'batch_size': {
                'values': [16, 32, 64, 128]
            },
            'learning_rate': {
                'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
            },
            'weight_decay': {
                'values': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
            },
            'num_epochs': {
                'values': [1000]
            },
            'model_type': {
                'values': ['ResNet18', 'ResNet34', 'ResNet50']
            },
        },
            'early_terminate': { # To reduce overfitting
            'type': 'hyperband',
            'min_iter': 400,
            'metric': 'val_loss'
            }
        }
    return sweep_config

def run_sweep():
    sweep_config = create_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="meal-portions-classification")

    # Start the sweep agent
    wandb.agent(sweep_id, function=train_classification_sweep, count=20)


if __name__ == "__main__":
    # csv_filepath = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\processed_portions_classification\processed_portions_classification.csv"
    # train_classification(csv_filepath, ResNet34(num_classes=3), 'portions_resnet34_batch64_lr0.001',
    #                      batch_size=64, num_epochs=450, learning_rate=0.001, weight_decay=0.0)
    run_sweep()