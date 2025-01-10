# Script for evaluation, such as portion size estimation and accuracy
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet18WithRGBandRGBD
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_model


def evaluate(model, test_loader,experiment_name, loss_fn, device):
    writer = SummaryWriter(f"runs-updated/{experiment_name}")
    model.to(device)
    all_true_labels = []
    all_predicted_labels = []
    total_loss = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Scale loss by batch size
            total_samples += labels.size(0)

            # Write current loss to TensorBoard
            writer.add_scalar('Loss/test', loss.item(), total_samples)

            # Store labels for plotting - matplotlib does not work with gpu
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(outputs.cpu().numpy())

    # Convert to numpy arrays because matplotlib cannot handle tensors
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)

    # Calculate metrics
    avg_loss = total_loss / total_samples
    mse = mean_squared_error(all_true_labels, all_predicted_labels)
    r2 = r2_score(all_true_labels, all_predicted_labels)

    # Write to Tensorboard
    writer.add_scalar('MSE/test', mse)
    writer.close()

    plt.figure(figsize=(12, 6))
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(all_true_labels, all_predicted_labels, alpha=0.6, color="purple")
    plt.plot([min(all_true_labels), max(all_true_labels)],
             [min(all_true_labels), max(all_true_labels)],
             color="red", linestyle="--", label="Ideal")
    plt.xlabel("True Glycemic Load Values")
    plt.ylabel("Predicted Glycemic Load Values")
    plt.title(f"Predicted vs True Values (R² = {r2:.4f})")
    plt.legend()

    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = all_predicted_labels - all_true_labels
    plt.scatter(all_predicted_labels, residuals, alpha=0.6, color="blue")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Glycemic Load Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.tight_layout()
    plt.savefig(f"{experiment_name}_regression_plots.png")
    plt.show()

    return mse, avg_loss


def plot_evaluation_metrics(true_labels, predicted_labels, experiment_name):
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot of predictions
    ax1.scatter(true_labels, predicted_labels, alpha=0.6, color="purple")
    ax1.plot([min(true_labels), max(true_labels)],
             [min(true_labels), max(true_labels)],
             color="red", linestyle="--", label="Ideal")
    ax1.set_xlabel("True Labels")
    ax1.set_ylabel("Predicted Labels")
    ax1.set_title("Predicted vs True Labels")
    ax1.legend()

    # Distribution of predictions
    ax2.hist(predicted_labels, bins=30, alpha=0.7, color='blue', label='Predicted')
    ax2.hist(true_labels, bins=30, alpha=0.7, color='red', label='True')
    ax2.set_xlabel("Label Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of True vs Predicted Labels")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"evaluation_plots_{experiment_name}.png")
    plt.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\outputs\models\not_pretrained_resnet18_batch32_lr0.00001\model.pth"
    trained_model = load_model(path, ResNet18WithRGBandRGBD, None, {'is_pretrained': False}, None, 'else')
    test_loader = torch.load(r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\outputs\models\not_pretrained_resnet18_batch32_lr0.00001\train_loader.pt")
    mse, avg_loss = evaluate(model=trained_model, test_loader=test_loader,
             experiment_name='untrained_resnet18_batch32_0.00001lr-new2', loss_fn=torch.nn.MSELoss(), device=device)
    print(f"MSE: {mse:.4f}, Average Loss: {avg_loss:.4f}")
