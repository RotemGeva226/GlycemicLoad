# Utility functions (e.g., for saving/loading models, metrics)
import torch

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

if __name__ == "__main__":
    load_model(
        model_path=r"/Portions Estimation/src/trained_models_regression\Test1.pth",
        model_class=ResNet34WithRGBandRGBD,
        model_args={'is_pretrained': False,}, optimizer_class=torch.optim.AdamW, optimizer_args={'lr': 1e-4})