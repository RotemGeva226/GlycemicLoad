import torchvision.models as models
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, type: int, num_classes: int, freeze_layers=False):
        super(Inception, self).__init__()

        inception_models = {
            3: models.inception_v3(pretrained=True)
        }

        self.inception = inception_models[type]
        self.inception.aux_logits = False

        if freeze_layers:
            for param in self.inception.parameters():
                param.requires_grad = False
                # Unfreeze the last fully connected (fc) layer
            in_features = self.inception.fc.in_features
            self.inception.fc = nn.Linear(in_features, num_classes)
            # Ensure the new fc layer is trainable
            for param in self.inception.fc.parameters():
                param.requires_grad = True

        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)

    def forward(self, input):
        out = self.inception(input).squeeze(dim=1)
        return out



