# The CNN models definition
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()

        # Load Resnet18 pretrained model
        self.resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        self.resnet.fc = nn.Linear(in_features=512,
                                   out_features=num_classes)  # Output glycemic load group (low, medium, high)

    def forward(self, x):
        # Pass the combined input through ResNet
        x = self.resnet(x)
        x = nn.Softmax(dim=1)(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        # Load Resnet18 pretrained model
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)  # Output glycemic load group (low, medium, high)


    def forward(self, x):
        # Pass the combined input through ResNet
        x = self.resnet(x)
        x = nn.Softmax(dim=1)(x)
        return x


if __name__ == '__main__':
    model = ResNet18WithRGBandRGBD()
    model.print()

