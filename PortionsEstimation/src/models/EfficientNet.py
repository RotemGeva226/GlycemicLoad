import torchvision.models as models
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self, type: int, input_channels: int, num_classes: int):
        super(EfficientNet, self).__init__()

        efficientnet_models = {
            0: models.efficientnet_b0(pretrained=True),
            3: models.efficientnet_b3(pretrained=True),
        }

        self.efficientnet = efficientnet_models[type]
        self.efficientnet.features[0][0] = nn.Conv2d(input_channels, self.efficientnet.features[0][0].out_channels,
                                                     kernel_size=(3, 3),
                                                     stride=(2, 2), padding=(1, 1),
                                                     bias=False)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, input):
        out = self.efficientnet(input).squeeze(dim=1)  # Tensor shape: (batch_size, 1) -> (batch_size)
        return out