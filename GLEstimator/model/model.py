import torch
from torchvision import transforms
from PIL import Image
from PortionsEstimation.src.models.EfficientNet import EfficientNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\GLEstimator\model\EfficientNet-B3.pth"

def load_model():
    model = EfficientNet(3, 3, 1)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to("cuda")
    model.eval()
    return model


def predict_portions(image_path: str) -> dict:
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to("cuda")  # [1, C, H, W]
    with torch.no_grad():
        output = model(input_tensor)
    portions = output.squeeze().tolist()
    return portions

if __name__ == "__main__":
    image_path = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\PortionsEstimation\src\data\raw\single_ingredient_images\dish_1556572657.jpg"
    result = predict_portions(image_path)
    print(result)