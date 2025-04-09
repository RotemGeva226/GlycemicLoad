import torchvision.models as models
import torch.nn as nn
import torch

EFFICIENT_NET_B1_WEIGHTS_PATH = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\PortionsEstimation\src\models\saved\efficientnet_b1_weights.pth"
class EfficientNet(nn.Module):
    def __init__(self, type: int, input_channels: int, num_classes: int,  freeze_layers=False):
        super(EfficientNet, self).__init__()

        if type == 1 or 2:
            match type:
                case 1:
                    self.efficientnet = models.efficientnet_b1()
                    self.efficientnet.load_state_dict(torch.load(EFFICIENT_NET_B1_WEIGHTS_PATH))
        else:
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

        if freeze_layers:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

            # Replace and unfreeze the final classifier layer
            for param in self.efficientnet.classifier[1].parameters():
                param.requires_grad = True

    def forward(self, input):
        out = self.efficientnet(input).squeeze(dim=1)  # Tensor shape: (batch_size, 1) -> (batch_size)
        return out