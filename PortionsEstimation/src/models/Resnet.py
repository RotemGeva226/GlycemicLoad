from tensorflow.python.ops import nn
import torchvision.models as models
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, num_layers: int, input_channels: int, num_classes: int, is_pretrained: bool):
        super(ResNet, self).__init__()

        resnet_models = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }

        if is_pretrained:
            weights = 'imagenet'

        self.resnet = resnet_models[num_layers](weights=weights)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_features=num_classes)

class ResNetRegression(ResNet):
    def forward(self, x):
        out = self.resnet(x).squeeze(dim=1)  # Tensor shape: (batch_size, 1) -> (batch_size)
        return out

class ResNetClassification(ResNet):
    def forward(self, x):
        x = self.resnet(x)
        x = nn.Softmax(dim=1)(x)
        return x


