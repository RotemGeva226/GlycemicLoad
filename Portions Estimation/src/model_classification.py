# The CNN models definition
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F


class ResNet34WithRGBandRGBD(nn.Module):
    def __init__(self, is_pretrained):
        super(ResNet34WithRGBandRGBD, self).__init__()

        # Load ResNet101 pretrained model
        self.resnet = models.resnet34(pretrained=is_pretrained)

        # Modify the first convolutional layer to accept 7 input channels (3 from RGB, 4 from RGBD)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Linear(in_features=512, out_features=1)  # Output glycemic load (single value)


    def forward(self, combined_input):
        # Pass the combined input through ResNet
        out = self.resnet(combined_input).squeeze(dim=1) # Tensor shape: (batch_size, 1) -> (batch_size)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        # Load Resnet18 pretrained model
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)  # Output glycemic load group (low, medium, high)


    def forward(self, x):
        # Pass the combined input through ResNet
        x = self.resnet(x)
        x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    model = ResNet18WithRGBandRGBD()
    model.print()

