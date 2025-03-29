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

def load_model(model_path, model_class, optimizer_class=None, model_args: dict=None, optimizer_args: dict=None, mode='continue_training'):
    try:
        model = model_class(**model_args)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if mode == 'continue_training':
            optimizer = optimizer_class(model.parameters(), **optimizer_args)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print('Model loaded successfully.')
            return model, optimizer, epoch, loss
        else:
            return model
    except Exception as e:
        print('An error occurred while loading the model:', e)

if __name__ == "__main__":
    load_model(
        model_path=r"/Portions Estimation/src/trained_models_regression\Test1.pth",
        model_class=ResNet34WithRGBandRGBD,
        model_args={'is_pretrained': False,}, optimizer_class=torch.optim.AdamW, optimizer_args={'lr': 1e-4})