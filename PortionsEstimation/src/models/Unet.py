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

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.unet(x)
        x = self.global_pool(x) # Reduces to shape [batch_size, channels, 1, 1]
        x = x.view(x.size(0), -1)  # Reshape to [batch_size, channels]
        return x
