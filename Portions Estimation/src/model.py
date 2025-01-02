# The CNN models definition
import torchvision.models as models
import torch.nn as nn


class ResNet101WithRGBandRGBD(nn.Module):
    def __init__(self):
        super(ResNet101WithRGBandRGBD, self).__init__()

        # Load ResNet101 pretrained model
        self.resnet = models.resnet101(pretrained=True)

        # Modify the first convolutional layer to accept 7 input channels (3 from RGB, 4 from RGBD)
        self.resnet.conv1 = nn.Conv2d(7, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Linear(in_features=2048, out_features=1)  # Output glycemic load (single value)

        print(self)


    def forward(self, combined_input):
        # Pass the combined input through ResNet
        out = self.resnet(combined_input)
        return out

