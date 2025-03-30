import torch
import numpy as np
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.Unet import UNet


def evaluate_regression(model, test_loader, experiment_name):
    """
    Evaluate the trained regression model on the test set.
    :param model: Trained model.
    :param test_loader: DataLoader for the test set.
    :param experiment_name: Name of the experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for tensor, mass in test_loader:
            tensor = tensor.float().to(device)
            mass = mass.float().to(device)

            outputs = model(tensor)

            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(mass.cpu().numpy().flatten())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    wandb.init(project="single-ingr-portions-regression", name=f"{experiment_name}-evaluation")
    wandb.log({
        "test_mse": mse,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2
    })

    wandb.log({
        "test_prediction_distribution": wandb.Histogram(all_predictions),
        "test_target_distribution": wandb.Histogram(all_targets),
        "test_error_distribution": wandb.Histogram(all_predictions - all_targets)
    })

    wandb.log({
        "test_predictions_vs_targets": wandb.plot.scatter(
            table=wandb.Table(data=[[x, y] for x, y in zip(all_targets, all_predictions)],
                              columns=["Actual", "Predicted"]),
            x="Actual",
            y="Predicted",
            title="Predicted vs. Actual Mass (Test Set)"
        )
    })

    wandb.finish()

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

if __name__ == "__main__":
    model_path = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\PortionsEstimation\src\models\saved\single_ingr_portions_regression\Unet_best_loss.pth"
    checkpoint = torch.load(model_path)
    model = UNet(3, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    tl_path = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\PortionsEstimation\src\models\saved\single_ingr_portions_regression\tl-Unet.pt"
    tl = torch.load(tl_path)
    evaluate_regression(model=model, test_loader=tl, experiment_name="EfficientNet-B0")
