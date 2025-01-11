# The CNN models definition
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet34_Weights, ResNet18_Weights


class ResNet101WithRGBandRGBD(nn.Module):
    def __init__(self):
        super(ResNet101WithRGBandRGBD, self).__init__()

        # Load ResNet101 pretrained model
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 7 input channels (3 from RGB, 4 from RGBD)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Linear(in_features=2048, out_features=1)  # Output glycemic load (single value)


    def forward(self, combined_input):
        # Pass the combined input through ResNet
        out = self.resnet(combined_input).squeeze(dim=1) # Outputs tensor shape: (batch_size, 1) -> (batch_size)
        return out

class ResNet50WithRGBandRGBD(nn.Module):
    def __init__(self, is_pretrained):
        super(ResNet50WithRGBandRGBD, self).__init__()

        # Load ResNet101 pretrained model
        self.resnet = models.resnet50(pretrained=is_pretrained)

        # Modify the first convolutional layer to accept 7 input channels (3 from RGB, 4 from RGBD)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Linear(in_features=2048, out_features=1)  # Output glycemic load (single value)


    def forward(self, combined_input):
        # Pass the combined input through ResNet
        out = self.resnet(combined_input).squeeze(dim=1) # Outputs tensor shape: (batch_size, 1) -> (batch_size)
        return out

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

class ResNet18WithRGBandRGBD(nn.Module):
    def __init__(self, is_pretrained):
        super(ResNet18WithRGBandRGBD, self).__init__()

        # Load ResNet101 pretrained model
        self.resnet = models.resnet18(pretrained=is_pretrained)

        # Modify the first convolutional layer to accept 7 input channels (3 from RGB, 4 from RGBD)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Linear(in_features=512, out_features=1)  # Output glycemic load (single value)


    def forward(self, combined_input):
        # Pass the combined input through ResNet
        out = self.resnet(combined_input).squeeze(dim=1) # Tensor shape: (batch_size, 1) -> (batch_size)
        return out

    def print(self):
        print(self.resnet)


# TODO: Add EfficientNet

if __name__ == '__main__':
    model = ResNet18WithRGBandRGBD()
    model.print()

