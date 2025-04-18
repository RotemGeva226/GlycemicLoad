import torch
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

model = torch.load("EfficientNet-B1.pth", map_location=torch.device("gpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def predict_portions(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]
    with torch.no_grad():
        output = model(input_tensor)
    portions = output.squeeze().tolist()
    return {"predicted_grams": portions}