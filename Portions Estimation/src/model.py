# The CNN models definition
import torchvision.models as models
import torch.nn as nn

def create_resnet101():
    resnet101 = models.resnet101(pretrained=True)
    resnet101.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False) # RGB+RGBD (7 channels total)
    num_features = resnet101.fc.in_features
    resnet101.fc = nn.Linear(num_features, 1)  # Predict glycemic load
    return resnet101

