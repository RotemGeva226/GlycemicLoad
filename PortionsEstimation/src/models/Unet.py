import segmentation_models_pytorch as smp
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNet, self).__init__()

        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels
        )

    def forward(self, x):
        x = self.unet(x)
        return x
